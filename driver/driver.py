class MMMDriver:
    def init_output(self) -> str:
        pass

    def ingest_data(self):
        pass

    def describe_data(self):
        pass

    def run_feature_engineering(self):
        pass

    def fit(self):
        pass

    def visualize(self):
        pass

    def save_model(self):
        pass

    def main(self, input_filename):
        self.init_output()
        self.ingest_data()
        self.run_feature_engineering()
        self.describe_data()
        self.fit()
        self.visualize()
        self.save_model()
