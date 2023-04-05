import os
import pandas as pd
import logging
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


class EvaluationMetrics:
    def __init__(self, data_path:str):
        self.current_path = os.getcwd()
        self.data = self.load_data(data_path)
        self.logger = self.custom_logger()

    def load_data(self, data_path) -> pd.DataFrame:
        """Function load the data and returns it as a dataframe"""
        return pd.read_json(data_path)

    def custom_logger(self):
        """Function that create logger with preset parameters for saving work information"""
        logging.basicConfig(filename='logs.log',
                            filemode='a',
                            level='DEBUG',
                            format='%(name)s %(levelname)s %(message)s')
        logger = logging.getLogger()
        return logger

    def calculate_metrics(self) -> None:
        """Calculate the f1 score and accuracy for the predicted values"""
        f1sc = f1_score(self.data['gt_corners'], self.data['rb_corners'], average='weighted')
        acc = accuracy_score(list(self.data['gt_corners']), list(self.data['rb_corners']), normalize=False)
        self.logger.debug(f'f1-score was successfully calculated. It is equal to {f1sc}')
        self.logger.debug(f'Accuracy was successfully calculated. It is equal to {acc}')
        print(classification_report(self.data['gt_corners'], self.data['rb_corners']))

    def draw_plots(self) -> None:
        """Function for drawing plots and saving them to the 'plots' folder in the current directory"""
        save_path = os.path.join(self.current_path, 'plots')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.figure(figsize=(16, 5))
        cf_matrix = confusion_matrix(list(self.data['gt_corners']), list(self.data['rb_corners']), labels=[4, 6, 8, 10])
        cf_plot = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d')
        cf_figure = cf_plot.get_figure()
        cf_figure.savefig(os.path.join(save_path, 'cf_matrix.png'), dpi=400)

    def pipeline(self) -> None:
        """Function runs metrics calculation and function to save plots"""
        self.calculate_metrics()
        self.draw_plots()
        self.logger.debug(f'Pipeline is finished.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the json file with data.")
    args = parser.parse_args()
    EvaluationMetrics(args.path).pipeline()