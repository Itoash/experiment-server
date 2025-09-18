from load_experiment import load_experiment_complete
import os
from dotenv import load_dotenv
load_dotenv()
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', '/test_data/')
if __name__ == "__main__":
    
    path = TEST_DATA_PATH + '2024_11_04-CTRL/'
    print(os.listdir(path))
    load_experiment_complete(path)
