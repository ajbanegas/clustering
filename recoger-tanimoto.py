import pandas as pd
import sys

def get_df(foo):
    df = pd.read_csv(foo, sep=' ', header=None)
    df.columns = ['tanimoto', 'ligand']
    return df

def format(n):
    return str(round(n, 3)).replace(".", ",")

if __name__ == "__main__":
    files = sys.argv[1:]

    df = None
    for foo in files:
        df_aux = get_df(foo)
        if df is None:
            df = df_aux
        else:
            df = pd.concat([df, df_aux], axis = 0)

    tc_mean = df['tanimoto'].mean()
    tc_max = df['tanimoto'].max()
    print(f"Files: {sys.argv[1]}, TC mean: {format(tc_mean)}, TC max: {format(tc_max)}")
