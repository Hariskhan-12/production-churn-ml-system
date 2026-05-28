from src.utils.config import ConfigLoader
from src.data.ingest import DataIngestion
from src.features.preprocess import DataPreprocessor

def main():

    config=ConfigLoader().load_config()

    #load_data
    df=DataIngestion(config).load_data()

    x,y,preproessor=DataPreprocessor(config).preprocess(df)


    print("X shape:", x.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()