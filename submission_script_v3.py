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

  import os
  import numpy as np
  import pandas as pd
  from scipy.optimize import nnls
  from sklearn.svm import NuSVR
  from sklearn.linear_model import Ridge
  from sklearn.preprocessing import StandardScaler

  # ---- HYPER-PARAMETERS ----
  EPS         = 1e-6
  CAL_EPS     = 1e-12
  N_RNA       = 1200
  N_MET       = 4000
  CAL_ALPHA   = 0.85
  RIDGE_ALPHA = 1e-2
  NU_SVR_NU   = 0.5
  NU_SVR_C    = 1.0
  USE_ENSEMBLE = True

  # ---- UTILITIES ----
  def normalize(P):
      P = P.clip(lower=EPS)
      P = P.div(P.sum(axis=0), axis=1)
      P = P.fillna(1.0 / P.shape[0])
      P = P.div(P.sum(axis=0), axis=1)
      return P

  def clr(P, eps=1e-12):
      X = np.log(np.clip(P.to_numpy(float), eps, None))
      X = X - X.mean(axis=0, keepdims=True)
      return X

  def inv_clr(X):
      Y = np.exp(X)
      Y = Y / (Y.sum(axis=0, keepdims=True) + 1e-12)
      return Y

  def geo_mean_proportions(prop_list, weights=None):
      """Average proportion DataFrames in CLR space (geometric mean)."""
      if weights is None:
          weights = [1.0 / len(prop_list)] * len(prop_list)
      log_sum = None
      for P, w in zip(prop_list, weights):
          L = np.log(np.clip(P.to_numpy(float), EPS, None))
          log_sum = L * w if log_sum is None else log_sum + L * w
      Y = np.exp(log_sum)
      Y = Y / (Y.sum(axis=0, keepdims=True) + 1e-12)
      return pd.DataFrame(Y, index=prop_list[0].index, columns=prop_list[0].columns)

  # ---- FEATURE SELECTION ----
  def select_features_specificity(ref_df, n_keep):
      """Rank genes by specificity x CV: favours marker genes."""
      if n_keep is None or n_keep >= ref_df.shape[0]:
          return ref_df.index
      R = ref_df.to_numpy(float)
      gene_max  = R.max(axis=1)
      gene_mean = R.mean(axis=1) + 1e-8
      gene_std  = R.std(axis=1)
      score = (gene_max / gene_mean) * (gene_std / gene_mean)
      return pd.Series(score, index=ref_df.index).sort_values(ascending=False).head(n_keep).index

  def select_features_fstat(ref_df, n_keep):
      """F-statistic style: between-group vs within-group variance."""
      if n_keep is None or n_keep >= ref_df.shape[0]:
          return ref_df.index
      R = ref_df.to_numpy(float)
      overall_mean = R.mean(axis=1, keepdims=True)
      between_var  = ((R - overall_mean) ** 2).mean(axis=1)
      within_var   = R.std(axis=1) + 1e-8
      score = between_var / within_var
      return pd.Series(score, index=ref_df.index).sort_values(ascending=False).head(n_keep).index

  # ---- PREPROCESSING ----
  def prep_align(mix_df, ref_df, n_keep=None, log1p=True, feature_method='specificity'):
      idx     = mix_df.index.intersection(ref_df.index)
      mix_df  = mix_df.loc[idx, :].astype(float)
      ref_df  = ref_df.loc[idx, :].astype(float)

      if feature_method == 'fstat':
          keep = select_features_fstat(ref_df, n_keep)
      else:
          keep = select_features_specificity(ref_df, n_keep)

      keep   = keep.intersection(mix_df.index)
      mix_df = mix_df.loc[keep, :]
      ref_df = ref_df.loc[keep, :]

      if log1p:
          mix_df = np.log1p(mix_df)
          ref_df = np.log1p(ref_df)

      scale  = ref_df.mean(axis=1).replace(0, np.nan).fillna(1.0)
      mix_df = mix_df.div(scale, axis=0)
      ref_df = ref_df.div(scale, axis=0)
      return mix_df, ref_df

  # ---- SOLVERS ----
  def nnls_deconv(mix_df, ref_df):
      R         = ref_df.to_numpy(dtype=float)
      celltypes = list(ref_df.columns)
      props, errs = [], []
      for s in mix_df.columns:
          y    = mix_df[s].to_numpy(dtype=float)
          p, _ = nnls(R, y)
          p    = np.maximum(p, 0.0) + EPS
          p   /= p.sum()
          err  = np.linalg.norm(R @ p - y) / (np.linalg.norm(y) + 1e-12)
          props.append(p)
          errs.append(err)
      P = pd.DataFrame(props, index=mix_df.columns, columns=celltypes).T
      E = pd.Series(errs, index=mix_df.columns)
      return P, E

  def nusvr_deconv(mix_df, ref_df):
      """
      CIBERSORT-style nu-SVR deconvolution with proper StandardScaler.
      Each gene (row) is scaled using fit statistics from the reference matrix.
      """
      R         = ref_df.to_numpy(dtype=float)
      celltypes = list(ref_df.columns)

      # Fit scaler on reference rows (genes x celltypes) -> transpose for sklearn
      scaler = StandardScaler()
      scaler.fit(R.T)          # scaler sees shape (n_celltypes, n_genes)
      R_sc = scaler.transform(R.T).T   # back to (n_genes, n_celltypes)

      props, errs = [], []
      for s in mix_df.columns:
          y    = mix_df[s].to_numpy(dtype=float)
          y_sc = (y - scaler.mean_) / (scaler.scale_ + 1e-8)

          try:
              svr = NuSVR(nu=NU_SVR_NU, kernel='linear', C=NU_SVR_C,
                          max_iter=30000, tol=1e-4)
              svr.fit(R_sc, y_sc)
              p = np.maximum(svr.coef_.flatten(), 0.0) + EPS
              p /= p.sum()
          except Exception:
              p, _ = nnls(R, y)
              p = np.maximum(p, 0.0) + EPS
              p /= p.sum()

          err = np.linalg.norm(R @ p - y) / (np.linalg.norm(y) + 1e-12)
          props.append(p)
          errs.append(err)

      P = pd.DataFrame(props, index=mix_df.columns, columns=celltypes).T
      E = pd.Series(errs, index=mix_df.columns)
      return P, E

  # ---- BASE PREDICTION ----
  def base_predict(mix_rna, ref_rna, mix_met, ref_met):
      # RNA
      mix_r, ref_r = prep_align(mix_rna, ref_rna, n_keep=N_RNA,
                                 log1p=True, feature_method='specificity')
      P_svr,  E_svr  = nusvr_deconv(mix_r, ref_r)
      P_nnls, E_nnls = nnls_deconv(mix_r, ref_r)

      if USE_ENSEMBLE:
          P_rna = geo_mean_proportions([P_svr, P_nnls], weights=[0.65, 0.35])
          E_rna = 0.65 * E_svr + 0.35 * E_nnls
      else:
          P_rna = P_svr
          E_rna = E_svr

      P_rna = normalize(P_rna)
      out   = P_rna.reindex(columns=mix_rna.columns)
      out   = normalize(out)

      if mix_met is None or ref_met is None:
          return out, E_rna

      # Methylation
      mix_m, ref_m = prep_align(mix_met, ref_met, n_keep=N_MET,
                                 log1p=False, feature_method='fstat')
      P_met_svr,  E_met_svr  = nusvr_deconv(mix_m, ref_m)
      P_met_nnls, E_met_nnls = nnls_deconv(mix_m, ref_m)

      if USE_ENSEMBLE:
          P_met = geo_mean_proportions([P_met_svr, P_met_nnls], weights=[0.65, 0.35])
          E_met = 0.65 * E_met_svr + 0.35 * E_met_nnls
      else:
          P_met = P_met_svr
          E_met = E_met_svr

      P_met = normalize(P_met)

      # Fuse RNA + methylation in log space (Aitchison-correct)
      common = out.columns.intersection(P_met.columns)
      if len(common) > 0:
          er   = E_rna.reindex(common).fillna(E_rna.max() if len(E_rna) else 1.0)
          em   = E_met.reindex(common).fillna(E_met.max() if len(E_met) else 1.0)
          wr   = 1.0 / (er + 1e-12)
          wm   = 1.0 / (em + 1e-12)
          wsum = wr + wm

          log_rna   = np.log(np.clip(out.loc[:, common].to_numpy(float), EPS, None))
          log_met   = np.log(np.clip(P_met.loc[:, common].to_numpy(float), EPS, None))
          log_fused = log_rna * (wr / wsum).to_numpy() + log_met * (wm / wsum).to_numpy()
          fused     = np.exp(log_fused)
          fused     = fused / (fused.sum(axis=0, keepdims=True) + 1e-12)
          out.loc[:, common] = pd.DataFrame(fused, index=out.index, columns=common)

      out = normalize(out)
      return out, E_rna

  # ---- CALIBRATION ----
  def calib_paths():
      p1 = os.path.join("attachement", "calibration_params.npz")
      p2 = os.path.join(os.path.dirname(__file__), "attachement", "calibration_params.npz")
      if os.path.exists(os.path.dirname(__file__)):
          return p1, p2
      return p1, p1

  def load_calibration():
      p1, p2 = calib_paths()
      path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
      if path is None:
          return None
      z = np.load(path, allow_pickle=False)
      return z["A"].astype(float), z["b"].astype(float)

  def save_calibration(A, b):
      p1, _ = calib_paths()
      folder = os.path.dirname(p1)
      if folder and not os.path.exists(folder):
          os.makedirs(folder, exist_ok=True)
      np.savez(p1, A=A, b=b)

  def train_and_save_calibration(ref_rna, ref_met):
      if not (os.path.isdir("data") and os.path.isdir("ground_truth")):
          return None

      import attachement.data_processing as dp

      celltypes = list(ref_rna.columns)
      allow = {"SBN5", "SDN5", "SDC5", "SDE5", "SDEL", "VITR"}
      X_list, Y_list = [], []

      for fn in sorted(os.listdir("data")):
          if not (fn.startswith("mixes_") and fn.endswith(".h5")):
              continue
          name = fn.replace("mixes_", "").replace(".h5", "")
          if name not in allow:
              continue
          mix_path = os.path.join("data", fn)
          gt_path  = os.path.join("ground_truth", "groundtruth_" + name + ".h5")
          if not os.path.exists(gt_path):
              continue

          mixes  = dp.read_hdf5(mix_path)
          gt_obj = dp.read_hdf5(gt_path)
          gt     = gt_obj.get("groundtruth", None)
          if gt is None:
              continue

          mix_rna = mixes["mix_rna"]
          mix_met = mixes.get("mix_met", None)

          if gt.shape[1] == mix_rna.shape[1]:
              gt.columns = list(mix_rna.columns)
          if not set(celltypes).issubset(set(gt.index)):
              continue

          gt5 = gt.loc[celltypes, :].astype(float).clip(lower=CAL_EPS)
          gt5 = gt5.div(gt5.sum(axis=0), axis=1)

          Pbase, _ = base_predict(mix_rna, ref_rna, mix_met, ref_met)
          Pbase    = Pbase.loc[celltypes, :].clip(lower=CAL_EPS)
          Pbase    = Pbase.div(Pbase.sum(axis=0), axis=1)

          X_list.append(clr(Pbase, eps=CAL_EPS).T)
          Y_list.append(clr(gt5,   eps=CAL_EPS).T)

      if not X_list:
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
      X    = clr(Pbase, eps=CAL_EPS)
      Y    = (A @ X) + b[:, None]
      Pcal = inv_clr(Y)
      Pcal = pd.DataFrame(Pcal, index=Pbase.index, columns=Pbase.columns)
      Pmix = CAL_ALPHA * Pcal + (1.0 - CAL_ALPHA) * Pbase
      return normalize(Pmix)

  # ---- MAIN ----
  mix_met = kwargs.get("mix_met", None)
  ref_met = kwargs.get("ref_met", None)

  out, _ = base_predict(mix, ref, mix_met, ref_met)

  calib = load_calibration()
  if calib is None:
      calib = train_and_save_calibration(ref, ref_met)

  if calib is not None:
      A, b = calib
      if A.shape == (out.shape[0], out.shape[0]) and b.shape[0] == out.shape[0]:
          out = apply_calibration(out, A, b)

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
            print("pip installation blocked by externally-managed environment.")
            print("Trying conda install: " + package_to_install)

            conda_ok = try_conda_install(package_to_install)

            if conda_ok:
                globals()[package] = importlib.import_module(package)
                continue
            print("Unable to install package " + package_to_install + " automatically.")

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

    cleaned_name=dataset_name.replace("mixes_", "").removesuffix(".h5")

    pred_prop = program(mixes_data["mix_rna"], reference_data["ref_bulkRNA"], mix_met=mixes_data["mix_met"], ref_met=reference_data["ref_met"]   )
    predi_dic[cleaned_name] = pred_prop

############################### 
### Code submission mode

# we generate a zip file with the 'program' source code

if not os.path.exists("submissions"):
    os.makedirs("submissions")

# we save the source code as a Python file named 'program.py':
# FIX: use utf-8 encoding to avoid cp1252 UnicodeEncodeError on Windows
with open(os.path.join("submissions", "program.py"), 'w', encoding='utf-8') as f:
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
