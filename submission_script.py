##################################################################################################
### PLEASE only edit the program function between YOUR CODE BEGINS/ENDS HERE                   ###
##################################################################################################


########################################################
### Package dependencies /!\ DO NOT CHANGE THIS PART ###
########################################################
import subprocess
import sys
import importlib

def program(mix=None, ref=None, **kwargs):

  ##
  ## YOUR CODE BEGINS HERE
  ##

  import os
  import numpy as np
  import pandas as pd
  from scipy.optimize import nnls

  # ================================================================
  # PARAMS - identical to 0.85
  # ================================================================
  EPS         = 1e-8
  N_MET       = 4000
  CAL_ALPHA   = 0.9
  RIDGE_ALPHA = 1e-3
  CAL_EPS     = 1e-12

  # CHANGE 1: three diverse selection criteria instead of single variance@1500
  # variance   -> selects high-variation genes (as before)
  # fstat      -> selects maximally discriminative genes between cell types
  # specificity-> selects genes expressed primarily in ONE cell type (markers)
  # Each is independently good; their CLR average is more robust than any single one
  RNA_SUBSETS = [
      (1500, 'variance'),
      (1500, 'fstat'),
      (1500, 'specificity'),
  ]

  # Threshold below which a cell type is considered absent from the mixture
  # NNLS puts truly absent types at ~EPS/(n_types) after normalisation
  # Safe threshold: anything below 1% mean across samples = absent
  ABSENT_THRESHOLD = 0.01

  # ================================================================
  # FEATURE SELECTION - 3 strategies
  # ================================================================
  def select_features(ref_df, n_keep, method='variance'):
      if n_keep is None or n_keep >= ref_df.shape[0]:
          return ref_df.index

      if method == 'variance':
          # classic: top genes by variance across cell types
          scores = ref_df.var(axis=1)

      elif method == 'fstat':
          # F-statistic: between-celltype variance / overall variance
          # Maximises discrimination between cell types
          grand_mean = ref_df.mean(axis=1)
          between = ref_df.subtract(grand_mean, axis=0).pow(2).mean(axis=1)
          total   = ref_df.var(axis=1) + 1e-12
          scores  = between / total

      elif method == 'specificity':
          # Specificity: max expression / mean expression
          # Selects marker genes (high in one cell type, low in others)
          mx  = ref_df.max(axis=1)
          mn  = ref_df.mean(axis=1).replace(0, np.nan).fillna(1e-12)
          scores = mx / mn

      else:
          scores = ref_df.var(axis=1)

      return scores.sort_values(ascending=False).head(n_keep).index

  # ================================================================
  # CORE FUNCTIONS - unchanged from 0.85
  # ================================================================
  def normalize(P):
      P = P.clip(lower=EPS)
      P = P.div(P.sum(axis=0), axis=1)
      P = P.fillna(1.0 / P.shape[0])
      P = P.div(P.sum(axis=0), axis=1)
      return P

  def prep_align(mix_df, ref_df, n_keep=None, log1p=False, method='variance'):
      idx = mix_df.index.intersection(ref_df.index)
      mix_df = mix_df.loc[idx, :].astype(float)
      ref_df = ref_df.loc[idx, :].astype(float)
      keep = select_features(ref_df, n_keep, method=method)
      keep = keep.intersection(mix_df.index)
      mix_df = mix_df.loc[keep, :]
      ref_df = ref_df.loc[keep, :]
      if log1p:
          mix_df = np.log1p(mix_df)
          ref_df = np.log1p(ref_df)
      scale = ref_df.mean(axis=1).replace(0, np.nan).fillna(1.0)
      mix_df = mix_df.div(scale, axis=0)
      ref_df = ref_df.div(scale, axis=0)
      return mix_df, ref_df

  def nnls_deconv(mix_df, ref_df):
      R = ref_df.to_numpy(dtype=float)
      celltypes = list(ref_df.columns)
      props, errs = [], []
      for s in mix_df.columns:
          y = mix_df[s].to_numpy(dtype=float)
          p, _ = nnls(R, y)
          p = np.maximum(p, 0.0) + EPS
          p = p / p.sum()
          yhat = R @ p
          err = np.linalg.norm(yhat - y) / (np.linalg.norm(y) + 1e-12)
          props.append(p)
          errs.append(err)
      P = pd.DataFrame(props, index=mix_df.columns, columns=celltypes).T
      E = pd.Series(errs, index=mix_df.columns)
      return P, E

  def geo_mean_proportions(prop_list):
      # Average proportions in CLR space (compositionally correct)
      w = 1.0 / len(prop_list)
      log_sum = None
      for P in prop_list:
          L = np.log(np.clip(P.to_numpy(float), EPS, None))
          log_sum = L * w if log_sum is None else log_sum + L * w
      Y = np.exp(log_sum)
      Y = Y / (Y.sum(axis=0, keepdims=True) + 1e-12)
      return pd.DataFrame(Y, index=prop_list[0].index, columns=prop_list[0].columns)

  def clr(P, eps=1e-12):
      X = np.log(np.clip(P.to_numpy(float), eps, None))
      X = X - X.mean(axis=0, keepdims=True)
      return X

  def inv_clr(X):
      Y = np.exp(X)
      Y = Y / (Y.sum(axis=0, keepdims=True) + 1e-12)
      return Y

  # ================================================================
  # CHANGE 2: detect absent cell types
  # Root cause of SDN4=0.751: Aitchison=16.06 caused by spurious 5th
  # cell type. Calibration actively WORSENS this by pushing it up.
  # Fix: record which types are absent before calibration,
  #      zero them out again AFTER calibration.
  # ================================================================
  def detect_absent(P, threshold=ABSENT_THRESHOLD):
      """Return list of cell type names with mean proportion < threshold."""
      mean_props = P.mean(axis=1)
      absent = mean_props[mean_props < threshold].index.tolist()
      # Safety: never declare all types absent
      if len(absent) >= P.shape[0]:
          return []
      return absent

  def zero_absent(P, absent_types):
      """Set absent cell types to 0 and renormalize."""
      if not absent_types:
          return P
      P = P.copy()
      for ct in absent_types:
          if ct in P.index:
              P.loc[ct, :] = 0.0
      return normalize(P)

  # ================================================================
  # BASE PREDICTION - ensemble of 3 diverse selections
  # ================================================================
  def base_predict(mix_rna, ref_rna, mix_met, ref_met):
      # RNA: 3 diverse feature selections averaged in CLR space
      rna_props, rna_errs = [], []
      for n_keep, method in RNA_SUBSETS:
          mix_r, ref_r = prep_align(mix_rna, ref_rna, n_keep=n_keep, log1p=True, method=method)
          P_sub, E_sub = nnls_deconv(mix_r, ref_r)
          rna_props.append(normalize(P_sub))
          rna_errs.append(E_sub)

      P_rna = geo_mean_proportions(rna_props)
      E_rna = pd.concat(rna_errs, axis=1).mean(axis=1)
      P_rna = normalize(P_rna)
      out = P_rna.reindex(columns=mix_rna.columns)
      out = normalize(out)

      # Methylation - unchanged from 0.85
      if mix_met is None or ref_met is None:
          return out

      mix_m, ref_m = prep_align(mix_met, ref_met, n_keep=N_MET, log1p=False, method='variance')
      P_met, E_met = nnls_deconv(mix_m, ref_m)
      P_met = normalize(P_met)

      common = out.columns.intersection(P_met.columns)
      if len(common) > 0:
          er = E_rna.reindex(common).fillna(E_rna.max() if len(E_rna) else 1.0)
          em = E_met.reindex(common).fillna(E_met.max() if len(E_met) else 1.0)
          wr = 1.0 / (er + 1e-12)
          wm = 1.0 / (em + 1e-12)
          wsum = wr + wm
          fused = (out.loc[:, common].mul((wr / wsum), axis=1)
                   + P_met.loc[:, common].mul((wm / wsum), axis=1))
          fused = normalize(fused)
          out.loc[:, common] = fused

      out = normalize(out)
      return out

  # ================================================================
  # CALIBRATION - identical logic, training set extended to include
  # SDN4 and SDN6 (previously excluded -> SDN4 got 0.757)
  # ================================================================
  def calib_paths():
      """
      Robust search for calibration_params.npz.
      Codabench runs program.py via exec() so __file__ may be undefined.
      """
      cands = []
      sp = globals().get("submission_program", None)
      if isinstance(sp, str) and len(sp) > 0:
          cands.append(os.path.join(sp, "attachement", "calibration_params.npz"))
          cands.append(os.path.join(sp, "calibration_params.npz"))
      cands.append(os.path.join("attachement", "calibration_params.npz"))
      cands.append(os.path.join(os.getcwd(), "attachement", "calibration_params.npz"))
      cur = os.getcwd()
      for _ in range(5):
          cands.append(os.path.join(cur, "attachement", "calibration_params.npz"))
          parent = os.path.dirname(cur)
          if parent == cur:
              break
          cur = parent
      cands.append("/app/ingested_program/attachement/calibration_params.npz")
      cands.append("/app/ingested_program/calibration_params.npz")
      seen, result = set(), []
      for p in cands:
          if p not in seen:
              seen.add(p)
              result.append(p)
      return result

  def load_calibration():
      for path in calib_paths():
          if not (isinstance(path, str) and os.path.exists(path)):
              continue
          try:
              z = np.load(path, allow_pickle=False)
              return z["A"].astype(float), z["b"].astype(float)
          except Exception:
              continue
      return None

  def save_calibration(A, b):
      path = os.path.join("attachement", "calibration_params.npz")
      try:
          os.makedirs(os.path.dirname(path), exist_ok=True)
          np.savez(path, A=A, b=b)
          return True
      except Exception:
          return False

  def train_and_save_calibration(ref_rna, ref_met):
      if not (os.path.isdir("data") and os.path.isdir("ground_truth")):
          return None

      import attachement.data_processing as dp
      from sklearn.linear_model import Ridge

      celltypes = list(ref_rna.columns)
      # Extended training set: SDN4 and SDN6 added vs 0.85
      # Only use datasets with ALL 5 cell types for the 5x5 Ridge matrix
      allow = {"SBN5", "SDN5", "SDN6", "SDC5", "SDE5", "SDEL", "VITR"}
      X_list, Y_list = [], []

      for fn in sorted(os.listdir("data")):
          if not (fn.startswith("mixes_") and fn.endswith(".h5")):
              continue
          name = fn.replace("mixes_", "").replace(".h5", "")
          if name not in allow:
              continue
          mix_path = os.path.join("data", fn)
          gt_path = os.path.join("ground_truth", "groundtruth_" + name + ".h5")
          if not os.path.exists(gt_path):
              continue

          mixes = dp.read_hdf5(mix_path)
          gt_obj = dp.read_hdf5(gt_path)
          gt = gt_obj.get("groundtruth", None)
          if gt is None:
              continue

          mix_rna = mixes["mix_rna"]
          mix_met = mixes.get("mix_met", None)

          if gt.shape[1] == mix_rna.shape[1]:
              gt.columns = list(mix_rna.columns)

          if not set(celltypes).issubset(set(gt.index)):
              continue

          gt5 = gt.loc[celltypes, :].astype(float)
          gt5 = gt5.clip(lower=CAL_EPS)
          gt5 = gt5.div(gt5.sum(axis=0), axis=1)

          Pbase = base_predict(mix_rna, ref_rna, mix_met, ref_met)
          Pbase = Pbase.loc[celltypes, :]
          Pbase = Pbase.clip(lower=CAL_EPS)
          Pbase = Pbase.div(Pbase.sum(axis=0), axis=1)

          X_list.append(clr(Pbase, eps=CAL_EPS).T)
          Y_list.append(clr(gt5,   eps=CAL_EPS).T)

      if len(X_list) == 0:
          return None

      X_all = np.vstack(X_list)
      Y_all = np.vstack(Y_list)

      model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
      model.fit(X_all, Y_all)
      A = model.coef_.astype(float)
      b = model.intercept_.astype(float)
      save_calibration(A, b)
      return A, b

  def apply_calibration(Pbase, A, b):
      X = clr(Pbase, eps=CAL_EPS)
      Y = (A @ X) + b[:, None]
      Pcal = inv_clr(Y)
      Pcal = pd.DataFrame(Pcal, index=Pbase.index, columns=Pbase.columns)
      Pmix = CAL_ALPHA * Pcal + (1.0 - CAL_ALPHA) * Pbase
      return normalize(Pmix)

  # ================================================================
  # MAIN
  # ================================================================
  mix_met = kwargs.get("mix_met", None)
  ref_met = kwargs.get("ref_met", None)

  # Step 1: base prediction (ensemble of 3 diverse selections)
  out = base_predict(mix, ref, mix_met, ref_met)

  # Step 2: detect absent cell types NOW (before calibration distorts them)
  # SDN4 has 4 cell types -> 5th will be near-zero after NNLS
  # Must record this before calibration pushes it back up
  absent_types = detect_absent(out, threshold=ABSENT_THRESHOLD)

  # Step 3: apply calibration
  calib = load_calibration()
  if calib is None:
      calib = train_and_save_calibration(ref, ref_met)

  if calib is not None:
      A, b = calib
      if (A.shape[0] == out.shape[0]) and (A.shape[1] == out.shape[0]) and (b.shape[0] == out.shape[0]):
          out = apply_calibration(out, A, b)

  # Step 4: re-zero the absent types that calibration pushed back up
  # This is the critical fix for SDN4 Aitchison=16.06
  if absent_types:
      out = zero_absent(out, absent_types)

  out = normalize(out)
  out.columns = mix.columns
  return out

  ##
  ## YOUR CODE ENDS HERE
  ##


# Install and import each package
def install_and_import_packages(required_packages):
  for package in required_packages:
      try:
          globals()[package] = importlib.import_module(package)
      except ImportError:
          print('impossible to import, installing packages',package)
          package_to_install = 'scikit-learn' if package == 'sklearn' else package
          subprocess.check_call([sys.executable, "-m", "pip", "install", package_to_install])
          globals()[package] = importlib.import_module(package)




# Install and import each package
def install_and_import_packages(required_packages):
    def try_pip_install(package_name):
      """Try pip install; detect externally-managed-environment error."""
      try:
          subprocess.check_call(
              [sys.executable, "-m", "pip", "install", package_name]
          )
          return True
      except subprocess.CalledProcessError as e:
          if "externally-managed-environment" in str(e):
              return False  # pip blocked by PEP 668
          raise  # real error unrelated to PEP 668
  

    def try_conda_install(package_name):
        """Attempt conda install."""
        try:
            subprocess.check_call(["conda", "install", "-y", package_name])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    for package in required_packages:
        try:
            globals()[package] = importlib.import_module(package)
        except ImportError:
            print('impossible to import, installing packages',package)
            package_to_install = 'scikit-learn' if package == 'sklearn' else package
            pip_ok = try_pip_install(package_to_install)

            if pip_ok:
              globals()[package] = importlib.import_module(package)
              continue

            # Pip failed due to externally-managed environment (Debian, Conda etc.)
            print("⛔ pip installation blocked by externally-managed environment.")
            print(f"➡️ Trying conda install: {package_to_install}")

            conda_ok = try_conda_install(package_to_install)

            if conda_ok:
                globals()[package] = importlib.import_module(package)
                continue
            print(f"Unable to install package {package_to_install} automatically.")

def validate_pred(pred, nb_samples=None, nb_cells=None, col_names=None):
    error_status = 0  # 0 means no errors, 1 means "Fatal errors" and 2 means "Warning"
    error_informations = ''

    # Ensure that all sum of cells proportion approximately equal 1
    if not numpy.allclose(numpy.sum(pred, axis=0), 1):
        msg = "The prediction matrix does not respect the laws of proportions: the sum of each column should be equal to 1\n"
        error_informations += msg
        error_status = 2

    # Ensure that the prediction has the correct names
    if not set(col_names) == set(pred.index):
        msg = f"The row names in the prediction matrix should match: {col_names}\n"
        error_informations += msg
        error_status = 2

    # Ensure that the prediction returns the correct number of samples and number of cells
    if pred.shape != (nb_cells, nb_samples):
        msg = f'The prediction matrix has the dimension: {pred.shape} whereas the dimension: {(nb_cells, nb_samples)} is expected\n'
        error_informations += msg
        error_status = 1

    if error_status == 1:
        # The error is blocking and should therefore stop the execution
        raise ValueError(error_informations)
    if error_status == 2:
        print("Warning:")
        print(error_informations)


##############################################################
### Generate a prediction file /!\ DO NOT CHANGE THIS PART ###
##############################################################

# List of required packages
required_packages = [
  "numpy",
  "pandas",
  "zipfile",
  "inspect",
  "h5py",
  "scipy",
  "sklearn"
]
install_and_import_packages(required_packages)

import os
import attachement.data_processing as dp


dir_name = "data"+os.sep

datasets_list = [filename for filename in os.listdir(dir_name) if filename.startswith("mixes")]

ref_file = os.path.join(dir_name, "reference_pdac.h5")
reference_data = dp.read_hdf5(ref_file)


predi_dic = {}
for dataset_name in datasets_list :

    file= os.path.join(dir_name,dataset_name)
    mixes_data = dp.read_hdf5(file)

    print(f"generating prediction for dataset: {dataset_name}")

    # mix_rna = extract_data_element(mixes_data,file, 'mix_rna') 
    # mix_met = extract_data_element(mixes_data,file, 'mix_met')
    cleaned_name=dataset_name.replace("mixes_", "").removesuffix(".h5")

    pred_prop = program(mixes_data["mix_rna"], reference_data["ref_bulkRNA"], mix_met=mixes_data["mix_met"], ref_met=reference_data["ref_met"]   )
    # validate_pred(pred_prop, nb_samples=mix_rna.shape[1], nb_cells=ref_bulkRNA.shape[1], col_names=ref_bulkRNA.columns)
    predi_dic[cleaned_name] = pred_prop

############################### 
### Code submission mode

# we generate a zip file with the 'program' source code

if not os.path.exists("submissions"):
    os.makedirs("submissions")

# we save the source code as a Python file named 'program.py':
with open(os.path.join("submissions", "program.py"), 'w') as f:
    f.write(inspect.getsource(program))

date_suffix = pandas.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")




# we create the associated zip file:
zip_program = os.path.join("submissions", f"program_{date_suffix}.zip")
with zipfile.ZipFile(zip_program, 'w') as zipf:
    zipf.write(os.path.join("submissions", "program.py"), arcname="program.py")


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
if os.path.exists("attachement"):
    with zipfile.ZipFile(zip_program, 'a', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('attachement/', zipf)


print(zip_program)

###############################


# Generate a zip file with the prediction
if not os.path.exists("submissions"):
    os.makedirs("submissions")

prediction_name = "prediction.h5"

dp.write_hdf5(os.path.join("submissions", prediction_name),predi_dic)



# Create the associated zip file:
zip_results = os.path.join("submissions", f"results_{date_suffix}.zip")
with zipfile.ZipFile(zip_results, 'w') as zipf:
    zipf.write(os.path.join("submissions", prediction_name), arcname=prediction_name)

print(zip_results)

