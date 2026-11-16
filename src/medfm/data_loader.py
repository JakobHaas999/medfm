"""Real-data loaders for the three medical tabular benchmarks.

Every loader returns ``(X, y, groups)``:

* ``X`` is a numeric ``float32`` feature matrix.
* ``y`` is the binary target.
* ``groups`` is the patient identifier used to keep a patient on one side of
  the outer and inner train/test splits.
"""

import pandas as pd
import numpy as np


def load_echo_data(path):
	"""Read the echocardiography data and predict report-coded LV systolic abnormality.

	This is a pre-echo prediction task: the model may only use information that
	is available before the echocardiogram is performed. We would like to
	investigate how well classification methods predict report-coded LV
	systolic abnormality in patients from information available before the
	echocardiogram.

	LV systolic abnormality (especially dysfunction) is common in septic shock
	(reported in 65% of patients in an early septic-shock series) and is often
	underestimated without echocardiography, which is the bedside tool used to
	characterise it (Vieillard-Baron, Ann Intensive Care 2011,
	DOI 10.1186/2110-5820-1-6).
	"""
	data1_path = path + "data1_echo_notes.pkl"
	df1 = pd.read_pickle(data1_path)

	# Step 1 -- define the target.
	#  0 as normal and -2, -1, 1, 2, 3 as (DOI 10.13026/xhrz-ht59)
	# abnormal. other codes excluded.
	NORMAL_VALUES = [0]
	ABNORMAL_VALUES = [-2, -1, 1, 2, 3]
	VALID_VALUES = NORMAL_VALUES + ABNORMAL_VALUES
	df_lv = df1[df1["LV_systolic"].isin(VALID_VALUES)].copy()
	TARGET = "y_LV_systolic_normal"
	df_lv[TARGET] = df_lv["LV_systolic"].isin(ABNORMAL_VALUES).astype("int8")

	# step 2 - shuffle patient order once, then keep subject_id as the grouping
	# key. Shuffling avoids accidentally processing all rows of one patient
	# together; the group key is also use in resampling fucntion to make 
    # sure we sample persons not rows
	rng = np.random.default_rng(42)
	order = {sid: i for i, sid in enumerate(rng.permutation(df_lv["subject_id"].drop_duplicates().to_numpy()))}
	df_lv = (df_lv.assign(_ord=df_lv["subject_id"].map(order))
               		.sort_values(["_ord", "subject_id"], kind="stable")
               		.drop(columns="_ord").reset_index(drop=True))

	# Patients, not rows, are the unit that must stay disjoint in resampling.
	groups = df_lv["subject_id"].reset_index(drop=True)

	# Step 3: remove everything that is not a pre-echo predictor.
	# The list below contains t all echocardiogram findings and identifiers and the target itself.
	drop_cols = [
			"LA_cavity", "RA_dilated", "LV_systolic", "LV_cavity", "LV_wall", "RV_cavity",
			"RV_systolic", "AV_stenosis", "MV_stenosis", "TV_regurgitation", "TV_stenosis",
			"TV_pulm_htn", "AV_regurgitation", "MV_regurgitation", "RA_pressure",
			"LV_diastolic", "RV_volume_overload", "RV_wall", "RV_pressure_overload",
    		"text", "row_id", "subject_id", "hadm_id", "category", "chartdate", 
      		"date_time", "test", "contrast", "technical_quality", "doppler", TARGET
	]
	df_clean = df_lv.drop(columns=drop_cols, errors="ignore").copy()

	# Step 4: encode the remaining pre-echo variables.
	# Numeric/bool columns are used as-is. Categorical columns are turn to binary dummy
	# features so the models receive only numbers. High-cardinality categories
	# are skipped to avoid creating hundreds of sparse dummy columns.
	num = df_clean.select_dtypes(include=["number", "bool"])
	cat = df_clean.select_dtypes(include=["object", "category"])
	high_card = [c for c in cat.columns if cat[c].nunique() > 30]
	cat_small = cat.drop(columns=high_card, errors="ignore")

	X_cat = pd.get_dummies(cat_small, drop_first=True)

	# Step 5: combine numeric and encoded columns. Drop constant columns
	# because they carry no information, keep missing values as NaN, and use
	# float32 for the model matrix.
	X_df1 = pd.concat([num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
	const_cols = [c for c in X_df1.columns if X_df1[c].nunique(dropna=True) <= 1]
	X_df1 = X_df1.drop(columns=const_cols, errors="ignore").astype("float32")

	# The target is aligned row-for-row with X and groups.
	y_df1 = df_lv[TARGET].astype(int).reset_index(drop=True)

	return X_df1, y_df1, groups



def load_glucose_data(path):

	"""Read the glucose data and build one row per eligible insulin event.


	"""
	data2_path = path + "data2_blood_glucose_management.pkl"
	df2 = pd.read_pickle(data2_path)

	df = df2.copy()

	# Step 1: parse the event, glucose, and pairing timestamps. Invalid values
	# are removed.
	for c in ["TIMER","STARTTIME","ENDTIME","GLCTIMER","GLCTIMER_AL"]:
		if c in df.columns:
			df[c] = pd.to_datetime(df[c], errors="coerce")

	# Step 2: separate insulin events from glucose readings.
	# An insulin event is a row with INPUT present and a valid start/end time.
	# It must also have a paired glucose reading measured BEFORE the insulin
	# starts.
	# A glucose reading is a row with GLC present and a valid measurement time.
	# Both are sorted inside their ICU stay so the pairing below cannot cross
	# from one ICU stay into another.
	stay_keys = ["SUBJECT_ID","ICUSTAY_ID"]
	df_ins = df.loc[
		df["INPUT"].notna()
		& df["STARTTIME"].notna()
		& df["ENDTIME"].notna()
		& (df["ENDTIME"] >= df["STARTTIME"])
		& df["GLCTIMER_AL"].notna()
		& df["GLC_AL"].notna()
		& (df["GLCTIMER_AL"] < df["STARTTIME"])
	].dropna(subset=stay_keys).sort_values(stay_keys+["STARTTIME"]).reset_index(drop=True)
	df_glc = df.loc[df["GLC"].notna()].dropna(subset=stay_keys+["GLCTIMER"]).sort_values(stay_keys+["GLCTIMER"]).reset_index(drop=True)

	# Step 3: pair each insulin event with the first glucose reading in the
	# SAME ICU stay that comes after ENDTIME. side="right" means the reading
	# must be strictly later, so a reading at the exact stop time is not used.
	# ENDTIME is the descriptor's "input stopped / rate changed" time, so the
	# outcome is measured after the delivered dose. The maximum follow-up also
	# starts at ENDTIME and depends on the insulin acting type:
	# short/regular 4-8 h, intermediate 10-12 h, long 12-24 h (DOI 10.13026/517s-2q57).  
 	# We use the upper bound of each range.
	# so a reading taken near the end of the insulin's effect is still counted.
	MAX_GAP_H = {"Short": 8.0, "Intermediate": 12.0, "Long": 24.0}
	DEFAULT_MAX_GAP_H = 8.0  # fallback when INSULINTYPE is missing
	glc_by_stay = dict(tuple(df_glc.groupby(stay_keys, sort=False)))
	pairs = []
	for key, ins in df_ins.groupby(stay_keys, sort=False):
		glc = glc_by_stay.get(key)
		if glc is None:
			continue
		pos = np.searchsorted(glc["GLCTIMER"].to_numpy(), ins["ENDTIME"].to_numpy(), side="right")
		ok = pos < len(glc)
		if not ok.any():
			continue
		ins = ins.loc[ok].copy()
		pos = pos[ok]
		gap_h = (glc["GLCTIMER"].to_numpy()[pos] - ins["ENDTIME"].to_numpy()) / np.timedelta64(1, "h")
		max_gap_h = ins["INSULINTYPE"].map(MAX_GAP_H).fillna(DEFAULT_MAX_GAP_H).to_numpy(dtype=float)
		keep = gap_h <= max_gap_h
		ins = ins.loc[keep].copy()
		pos = pos[keep]
		if not len(ins):
			continue
		ins["GLC_next"] = glc["GLC"].to_numpy(dtype=float)[pos]
		ins["gap_h"] = gap_h[keep]
		ins["_reading"] = glc.index.to_numpy()[pos]
		pairs.append(ins)

	# Step 4: keep one row per glucose reading.
	# Several insulin events can point to the same reading. Keep only the event
	# closest to that reading (smallest gap_h), so the same outcome is not
	# counted multiple times. 
	df_merged2 = pd.concat(pairs, ignore_index=True)
	df_merged2 = (df_merged2.sort_values("gap_h", kind="stable")
                             .drop_duplicates("_reading", keep="first")
                             .drop(columns=["gap_h","_reading"])
                             .reset_index(drop=True))

	# Step 5: define the target as inclusive Time in Range (70-180 mg/dL).
	# The target is the paired (((next))) reading.
	TARGET = "y_glucose_normal"
	df_merged2[TARGET] = df_merged2["GLC_next"].between(70, 180).astype("int8") 

	# Step 6: shuffle patient order and keep SUBJECT_ID as the grouping key.
	# Needed for resampling, for the split.
	rng = np.random.default_rng(42)
	order = {sid: i for i, sid in enumerate(rng.permutation(df_merged2["SUBJECT_ID"].drop_duplicates().to_numpy()))}
	df_merged2 = (df_merged2.assign(_ord=df_merged2["SUBJECT_ID"].map(order))
                         .sort_values(["_ord","STARTTIME"], kind="stable")
                         .drop(columns="_ord").reset_index(drop=True))

	# Step 7: keep only predictors known at the insulin event.
  
	# INPUT is the bolus dose; for infusions it is rate x duration and therefore
	# future information, so infusions receive a structural 0 instead. INPUT_HRS
	# is the infusion rate; boluses have no infusion rate, so they receive a
	# structural 0.  Genuine missing values stay NaN.
	is_infusion = df_merged2["EVENT"].eq("INFUSION")
	df_merged2["bolus_dose_u"] = np.where(is_infusion, 0.0, df_merged2["INPUT"].astype(float))
	df_merged2["infusion_rate_u_hr"] = np.where(is_infusion, df_merged2["INPUT_HRS"].astype(float), 0.0)
	df_merged2["glucose_before_insulin"] = df_merged2["GLC_AL"].astype(float)
	df_merged2["minutes_since_glucose"] = (
		(df_merged2["STARTTIME"] - df_merged2["GLCTIMER_AL"]).dt.total_seconds() / 60.0
	)

	predictors = [
		"first_ICU_stay", "bolus_dose_u", "infusion_rate_u_hr",
		"INSULINTYPE", "EVENT", "glucose_before_insulin", "minutes_since_glucose",
	]

	# Mark the two categorical predictors so they are encoded as dummy variables.
	for c in ["INSULINTYPE","EVENT"]:
		df_merged2[c] = df_merged2[c].astype("category")

	# Step 8: build and clean the feature matrix.
	# Numeric/bool columns are used as-is. Categorical columns as described.
	# Constant columns are dropped, remaining missing values stay NaN.
	X_raw = df_merged2[predictors].reset_index(drop=True)

	num = X_raw.select_dtypes(include=["number", "bool"])
	cat = X_raw.select_dtypes(include=["object", "category"])
	X_cat = pd.get_dummies(cat, drop_first=True)
	X_df2 = pd.concat([num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
	const_cols = [c for c in X_df2.columns if X_df2[c].nunique(dropna=True) <= 1]
	X_df2 = X_df2.drop(columns=const_cols)
	X_df2 = X_df2.astype("float32")

	# The target and patient groups are aligned row-for-row with X.
	y_df2 = df_merged2[TARGET].astype(int).reset_index(drop=True)
	groups = df_merged2["SUBJECT_ID"].reset_index(drop=True)

	return X_df2, y_df2, groups





def load_blood_gas_data(path):
	"""Read blood-gas data and predict whether SaO2 is below 88%.

	The research question is whether this hypoxemia
	label is predictable from SpO2 and the other clinical variables available
	at the blood gas. SpO2 is kept as a predictor.
	"""
	data3_path = path + "data3_blood_gas_oximetry.pkl" 
	df3 = pd.read_pickle(data3_path)

	# Step 1: label hypoxemia. SaO2 below 88% is class 1; 88% or above
	# is class 0. The cutoff comes from the pulse-oximetry bias literature
	# (Martins et al., 2024, https://doi.org/10.48550/arXiv.2408.04396. 
 	# The benchmark is	# "hypoxemia or not" label. 
	THRESH_SAO2 = 88.0
	assert 'SaO2' in df3.columns, "SaO2 column missing"
	df_temp = df3.copy()
	df_temp['y'] = (df_temp['SaO2'] < THRESH_SAO2).astype(int)

	# Step 2: shuffle patient order and keep unique_subject_id as the grouping
	# key. Patients, not blood-gas rows.
	rng = np.random.default_rng(42)
	order = {sid: i for i, sid in enumerate(rng.permutation(df_temp['unique_subject_id'].drop_duplicates().to_numpy()))}
	df_temp = (df_temp.assign(_ord=df_temp['unique_subject_id'].map(order))
                 .sort_values(['_ord', 'unique_subject_id'], kind='stable')
                 .drop(columns='_ord')
                 .reset_index(drop=True))

	groups = df_temp['unique_subject_id']

	# Step 3: remove every column that is not available before the prediction.  (DOI 10.13026/phvt-3277)
	abg_cols = ['SaO2', 'pO2', 'pCO2', 'pH', 'Carboxyhemoglobin', 'Methemoglobin']
	outcome_cols = ['in_hospital_mortality', 'los_hospital', 'los_ICU']
	time_cols = [c for c in df_temp.columns if 'timestamp' in c.lower() or c.startswith('datetime_')]
	future_cols = [c for c in df_temp.columns if 'sofa_future_' in c]
	id_cols = ['unique_subject_id', 'unique_hospital_admission_id', 'unique_icustay_id',
           'subject_id', 'hospital_admission_id', 'icustay_id', 'hospitalid']

	drop_cols = set(abg_cols + outcome_cols + time_cols + future_cols + id_cols + ['y'])  # 'y' is the target

	# Step 4: build and clean the feature matrix.
	# Numeric AND bool columns are kept. Categorical columns are turned to dummy,
	# constant columns are dropped, and missing values stay NaN.
	# leaves no future SOFA columns.
	keep_df = df_temp.drop(columns=list(drop_cols), errors='ignore')
	num = keep_df.select_dtypes(include=['number', 'bool'])
	cat = keep_df.select_dtypes(include=['object', 'category'])

	X_df3 = pd.concat([num, pd.get_dummies(cat, drop_first=True)], axis=1).astype('float32')

	const_cols = [c for c in X_df3.columns if X_df3[c].nunique(dropna=True) <= 1]
	X_df3 = X_df3.drop(columns=const_cols, errors='ignore')

	# The target is aligned row-for-row with X and groups.
	y_df3 = df_temp['y'].astype(int)

	return X_df3, y_df3, groups