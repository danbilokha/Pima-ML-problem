import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler


class PreparedDataset:
    def data_cleaning(self):
        # Calculate the median value for BMI
        median_bmi = self.diabetes_dataset['BMI'].median()
        # Substitute it in the BMI column of the
        # dataset where values are 0
        self.diabetes_dataset['BMI'] = self.diabetes_dataset['BMI'].replace(
            to_replace=0, value=median_bmi)

        # Calculate the median value for BloodPressure
        median_bloodp = self.diabetes_dataset['BloodPressure'].median()
        # Substitute it in the BloodP column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['BloodPressure'] = self.diabetes_dataset['BloodPressure'].replace(
            to_replace=0, value=median_bloodp)

        # Calculate the median value for Glucose
        median_plglcconc = self.diabetes_dataset['Glucose'].median()
        # Substitute it in the PlGlcConc column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['Glucose'] = self.diabetes_dataset['Glucose'].replace(
            to_replace=0, value=median_plglcconc)

        # Calculate the median value for SkinThickness
        median_skinthick = self.diabetes_dataset['SkinThickness'].median()
        # Substitute it in the SkinThickness column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['SkinThickness'] = self.diabetes_dataset['SkinThickness'].replace(
            to_replace=0, value=median_skinthick)

        # Calculate the median value for TwoHourSerIns
        median_twohourserins = self.diabetes_dataset['Insulin'].median()
        # Substitute it in the TwoHourSerIns column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['Insulin'] = self.diabetes_dataset['Insulin'].replace(
            to_replace=0, value=median_twohourserins)

    def feature_scaling(self):
        scaler = Scaler()
        scaler.fit(self.train_dataset)

        # print(self.train_dataset, self.test_dataset)

        self.train_set_scaled = pd.DataFrame(scaler.transform(self.train_dataset), index=self.train_dataset.index,
                                             columns=self.train_dataset.columns)
        self.test_set_scaled = pd.DataFrame(scaler.transform(self.test_dataset), index=self.test_dataset.index,
                                            columns=self.test_dataset.columns)

        self.train_X = self.train_set_scaled[self.train_set_scaled.columns[:8]]
        self.train_Y = self.train_set_scaled['Outcome']

        self.test_X = self.test_set_scaled[self.test_set_scaled.columns[:8]]
        self.test_Y = self.test_set_scaled['Outcome']

    def __init__(self, diabetes_dataset_path='data/diabetes.csv'):
        self.dataset_path = diabetes_dataset_path
        self._load_dataset()
        self._split_dataset_train_test()

    def _load_dataset(self):
        self.diabetes_dataset = pd.read_csv(self.dataset_path)

    def _split_dataset_train_test(self):
        train, test = train_test_split(self.diabetes_dataset, test_size=0.20, random_state=0,
                                       stratify=self.diabetes_dataset['Outcome'])

        self.train_dataset = train
        self.test_dataset = test

        self.train_X = train[train.columns[:8]]
        self.train_Y = train['Outcome']

        self.test_X = test[test.columns[:8]]
        self.test_Y = test['Outcome']
