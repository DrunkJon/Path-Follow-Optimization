import pandas as pd
import numpy as np


def main():
    df = pd.DataFrame(
        {
            "robo": [np.array([100.0,500.0,2.5])],
            "goal": [np.array([50.0,250.0])],
        },
        index=[0]
    )
    print(df)
    cols = df.columns
    for i in range(1,51):
        df.loc[i] = {
            "robo": np.random.randn(3).round(1),
            "goal": np.random.randn(2).round(1),
            }
    print(df)


if __name__ == '__main__':
    main() 