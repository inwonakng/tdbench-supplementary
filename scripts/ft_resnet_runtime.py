import pandas as pd


ori_df = pd.read_csv("data_mode_switch_results_w_reg.csv")

df = ori_df[
    (ori_df["Subset"] == "Train")
    & (ori_df["Default Train Time"] != -1)
    & (ori_df["Inference Time"] != -1)
]
df["Identifier"] = df["Dataset"] + " " + df["Data Mode"] + " " + df["N"].astype(str)
resnet = df[df["Classifier"] == "ResNet"]
ft = df[df["Classifier"] == "FTTransformer"]

resnet_match = df[df["Identifier"].isin(resnet["Identifier"])]
ft_match = df[df["Identifier"].isin(ft["Identifier"])]


res_comps = pd.DataFrame(
    [
        {
            "Classifier": clf,
            "Train Time": by_clf["Default Train Time"].mean(),
            "Inference Time": by_clf["Inference Time"].mean(),
        }
        for clf, by_clf in resnet_match.groupby("Classifier")
    ]
)


ori_perfs = []
for ds, by_ds in resnet.groupby("Dataset"):
    ori = ori_df[
        (ori_df["Dataset"] == ds)
        & (ori_df["Data Mode"] == "Mixed -> Original -> Mixed")
        & (ori_df["N"] == 100)
        & (ori_df["Subset"] == "Test")
    ]
    ori_perfs.append({clf: by_clf["Score"].mean() for clf, by_clf in ori.groupby("Classifier")})
