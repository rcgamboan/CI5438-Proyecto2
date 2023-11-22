import pandas as pd

def main():

    with open("../docs/spambase.names", "r") as f:
        column_names = f.readlines()
    
    nombres = []
    nombres = column_names[33:]
    for i, name in enumerate(nombres):
        nombres[i] = name.split(":")[0]
    print(nombres)
    nombres.append("spam")

    df = pd.read_csv("../docs/spambase.data", names=nombres)
    print(df.tail())

if __name__ == "__main__":
    main()