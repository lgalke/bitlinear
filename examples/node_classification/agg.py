import pandas as pd
df = pd.read_csv("examples/node_classification/results.csv")

df_grouped = df.groupby(["Dataset", "Model"])
acc_mean = df_grouped["Accuracy"].mean()
acc_std = df_grouped["Accuracy"].std()

acc_sem = acc_std / (df_grouped["Accuracy"].count() ** 0.5)

print("Count", df_grouped["Accuracy"].count())
print("Mean", acc_mean)
print("SD", acc_std)
print("SEM", acc_sem)
print("1.96 * SEM", 1.96 * acc_sem)
