import zipfile
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

caminho_zip = "data/detalhe_votacao_munzona_2026.zip"
arquivo_brasil = "detalhe_votacao_munzona_2026_BRASIL.csv"

with zipfile.ZipFile(caminho_zip, "r") as arquivo_zip:
    with arquivo_zip.open(arquivo_brasil) as arquivo_csv:
        dados = pd.read_csv(
            arquivo_csv,
            sep=";",
            encoding="latin-1"
        )

print("Quantidade de linhas:", len(dados))
print("Quantidade de colunas:", len(dados.columns))

print("\nPrimeiras 5 linhas:")
print(dados.head())

print("\nColunas disponíveis:")

for coluna in dados.columns:
    print(coluna)

print("\nExemplo dos dados que vamos usar:")

colunas_principais = [
    "SG_UF",
    "NM_MUNICIPIO",
    "NR_ZONA",
    "DS_CARGO",
    "QT_APTOS",
    "QT_COMPARECIMENTO",
    "QT_ABSTENCOES",
    "QT_VOTOS",
    "QT_TOTAL_VOTOS_VALIDOS",
    "QT_VOTOS_BRANCOS",
    "QT_TOTAL_VOTOS_NULOS"
]

print(dados[colunas_principais].head(10).to_string(index=False))

print("\nCargos encontrados:")
print(dados["DS_CARGO"].unique())

print("\nTurnos encontrados:")
print(dados["NR_TURNO"].unique())

dados_presidente = dados[dados["DS_CARGO"] == "Presidente"].copy()

print("\nQuantidade de registros de Presidente:")
print(len(dados_presidente))

contagem_por_municipio = (
    dados_presidente
    .groupby(["SG_UF", "NM_MUNICIPIO"])
    .size()
    .sort_values(ascending=False)
)

print("\nMunicípios com mais registros:")
print(contagem_por_municipio.head(10))

dados_presidente_brasil = dados_presidente[
    dados_presidente["SG_UF"] != "ZZ"
].copy()

print("\nRegistros de Presidente no Brasil:")
print(len(dados_presidente_brasil))

municipios = (
    dados_presidente_brasil
    .groupby(
        ["SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO"],
        as_index=False
    )
    .agg(
        QT_APTOS=("QT_APTOS", "sum"),
        QT_COMPARECIMENTO=("QT_COMPARECIMENTO", "sum"),
        QT_ABSTENCOES=("QT_ABSTENCOES", "sum"),
        QT_VOTOS=("QT_VOTOS", "sum"),
        QT_VOTOS_VALIDOS=("QT_TOTAL_VOTOS_VALIDOS", "sum"),
        QT_VOTOS_BRANCOS=("QT_VOTOS_BRANCOS", "sum"),
        QT_VOTOS_NULOS=("QT_TOTAL_VOTOS_NULOS", "sum")
    )
)

print("\nQuantidade de municípios:")
print(len(municipios))

print("\nExemplo - Recife:")
print(
    municipios[
        (municipios["SG_UF"] == "PE") &
        (municipios["NM_MUNICIPIO"] == "RECIFE")
    ].to_string(index=False)
)

municipios["TX_ABSTENCAO"] = (
    municipios["QT_ABSTENCOES"]
    / municipios["QT_APTOS"]
) * 100

print("\nResumo da taxa de abstenção:")

print("Média:", municipios["TX_ABSTENCAO"].mean())
print("Mediana:", municipios["TX_ABSTENCAO"].median())
print("Mínimo:", municipios["TX_ABSTENCAO"].min())
print("Máximo:", municipios["TX_ABSTENCAO"].max())

menor_abstencao = municipios.loc[
    municipios["TX_ABSTENCAO"].idxmin()
]

maior_abstencao = municipios.loc[
    municipios["TX_ABSTENCAO"].idxmax()
]

print("\nMenor taxa de abstenção:")
print(
    menor_abstencao[
        ["SG_UF", "NM_MUNICIPIO", "QT_APTOS", "TX_ABSTENCAO"]
    ]
)

print("\nMaior taxa de abstenção:")
print(
    maior_abstencao[
        ["SG_UF", "NM_MUNICIPIO", "QT_APTOS", "TX_ABSTENCAO"]
    ]
)

municipios["FAIXA_ELEITORADO"] = pd.cut(
    municipios["QT_APTOS"],
    bins=[0, 10000, 50000, 100000, 500000, float("inf")],
    labels=[
        "Até 10 mil",
        "10 a 50 mil",
        "50 a 100 mil",
        "100 a 500 mil",
        "Acima de 500 mil"
    ]
)

print("\nQuantidade de municípios por faixa de eleitorado:")
print(municipios["FAIXA_ELEITORADO"].value_counts().sort_index())

resumo_por_faixa = (
    municipios
    .groupby("FAIXA_ELEITORADO", observed=True)["TX_ABSTENCAO"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\nResumo da abstenção por faixa de eleitorado:")
print(resumo_por_faixa)

pequenos = municipios[
    municipios["FAIXA_ELEITORADO"] == "Até 10 mil"
]

q1 = pequenos["TX_ABSTENCAO"].quantile(0.25)
q3 = pequenos["TX_ABSTENCAO"].quantile(0.75)

iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

print("\nIQR - Municípios com até 10 mil eleitores:")
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Limite inferior:", limite_inferior)
print("Limite superior:", limite_superior)

outliers_pequenos = pequenos[
    (pequenos["TX_ABSTENCAO"] < limite_inferior) |
    (pequenos["TX_ABSTENCAO"] > limite_superior)
]

print("\nQuantidade de outliers:")
print(len(outliers_pequenos))

print("\nMunicípios identificados como outliers:")
print(
    outliers_pequenos[
        ["SG_UF", "NM_MUNICIPIO", "QT_APTOS", "TX_ABSTENCAO"]
    ]
    .sort_values("TX_ABSTENCAO")
    .to_string(index=False)
)

print("\nOutliers por estado:")

print(
    outliers_pequenos["SG_UF"]
    .value_counts()
)

total_pequenos_por_estado = (
    pequenos["SG_UF"]
    .value_counts()
)

outliers_por_estado = (
    outliers_pequenos["SG_UF"]
    .value_counts()
)

analise_estados = pd.DataFrame({
    "TOTAL_MUNICIPIOS": total_pequenos_por_estado,
    "OUTLIERS": outliers_por_estado
}).fillna(0)

analise_estados["PERCENTUAL_OUTLIERS"] = (
    analise_estados["OUTLIERS"]
    / analise_estados["TOTAL_MUNICIPIOS"]
) * 100

analise_estados = analise_estados.sort_values(
    "PERCENTUAL_OUTLIERS",
    ascending=False
)

print("\nPercentual de outliers por estado:")
print(analise_estados)

def detectar_outliers_iqr(grupo):
    q1 = grupo["TX_ABSTENCAO"].quantile(0.25)
    q3 = grupo["TX_ABSTENCAO"].quantile(0.75)

    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = grupo[
        (grupo["TX_ABSTENCAO"] < limite_inferior) |
        (grupo["TX_ABSTENCAO"] > limite_superior)
    ].copy()

    return outliers

outliers_todas_faixas = []

for faixa in municipios["FAIXA_ELEITORADO"].cat.categories:
    grupo = municipios[
        municipios["FAIXA_ELEITORADO"] == faixa
    ]

    outliers = detectar_outliers_iqr(grupo)

    outliers["FAIXA_ELEITORADO"] = faixa

    outliers_todas_faixas.append(outliers)

todos_outliers = pd.concat(
    outliers_todas_faixas,
    ignore_index=True
)

print("\nQuantidade de outliers por faixa:")
print(
    todos_outliers["FAIXA_ELEITORADO"]
    .value_counts()
    .sort_index()
)

total_por_faixa = (
    municipios["FAIXA_ELEITORADO"]
    .value_counts()
    .sort_index()
)

outliers_por_faixa = (
    todos_outliers["FAIXA_ELEITORADO"]
    .value_counts()
    .reindex(total_por_faixa.index, fill_value=0)
)

percentual_por_faixa = (
    outliers_por_faixa
    / total_por_faixa
) * 100

resumo_outliers = pd.DataFrame({
    "TOTAL_MUNICIPIOS": total_por_faixa,
    "OUTLIERS": outliers_por_faixa,
    "PERCENTUAL_OUTLIERS": percentual_por_faixa
})

print("\nResumo de outliers por faixa:")
print(resumo_outliers)


municipios["Z_SCORE"] = municipios.groupby(
    "FAIXA_ELEITORADO",
    observed=True
)["TX_ABSTENCAO"].transform(
    lambda x: (x - x.mean()) / x.std()
)

outliers_zscore = municipios[
    municipios["Z_SCORE"].abs() > 3
]

print("\nOutliers pelo Z-score:")
print(len(outliers_zscore))

municipios["TX_BRANCOS"] = (
    municipios["QT_VOTOS_BRANCOS"]
    / municipios["QT_VOTOS"]
) * 100

municipios["TX_NULOS"] = (
    municipios["QT_VOTOS_NULOS"]
    / municipios["QT_VOTOS"]
) * 100

variaveis = municipios[
    ["QT_APTOS", "TX_ABSTENCAO", "TX_BRANCOS", "TX_NULOS"]
]

modelo = IsolationForest(
    contamination=0.02,
    random_state=42
)

municipios["ISOLATION_FOREST"] = modelo.fit_predict(variaveis)

outliers_isolation = municipios[
    municipios["ISOLATION_FOREST"] == -1
]

print("\nOutliers pelo Isolation Forest:")
print(len(outliers_isolation))

municipios["OUTLIER_IQR"] = municipios["CD_MUNICIPIO"].isin(
    todos_outliers["CD_MUNICIPIO"]
)

municipios["OUTLIER_ZSCORE"] = municipios["CD_MUNICIPIO"].isin(
    outliers_zscore["CD_MUNICIPIO"]
)

municipios["OUTLIER_ISOLATION"] = (
    municipios["ISOLATION_FOREST"] == -1
)

municipios["TOTAL_METODOS"] = (
    municipios["OUTLIER_IQR"].astype(int)
    + municipios["OUTLIER_ZSCORE"].astype(int)
    + municipios["OUTLIER_ISOLATION"].astype(int)
)

print("\nMunicípios identificados por 2 ou mais métodos:")

resultado_consenso = municipios[
    municipios["TOTAL_METODOS"] >= 2
]

print(len(resultado_consenso))

print("\nQuantidade de municípios por número de métodos:")
print(resultado_consenso["TOTAL_METODOS"].value_counts().sort_index())

consenso_total = municipios[
    municipios["TOTAL_METODOS"] == 3
]

print("\nMunicípios identificados pelos 3 métodos:")

print(
    consenso_total[
        [
            "SG_UF",
            "NM_MUNICIPIO",
            "QT_APTOS",
            "TX_ABSTENCAO",
            "TX_BRANCOS",
            "TX_NULOS"
        ]
    ]
    .sort_values("TX_ABSTENCAO", ascending=False)
    .to_string(index=False)
)

plt.figure(figsize=(10, 6))

plt.scatter(
    municipios["QT_APTOS"],
    municipios["TX_ABSTENCAO"],
    alpha=0.3,
    label="Demais municípios"
)

plt.scatter(
    consenso_total["QT_APTOS"],
    consenso_total["TX_ABSTENCAO"],
    marker="x",
    s=100,
    label="Identificados pelos 3 métodos"
)

plt.xscale("log")

plt.xlabel("Quantidade de eleitores aptos")
plt.ylabel("Taxa de abstenção (%)")
plt.title("Taxa de abstenção por tamanho do eleitorado")

plt.legend()
plt.grid(alpha=0.2)

plt.legend()
plt.grid(alpha=0.2)

plt.savefig(
    "outputs/grafico_dispersao_abstencao.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()





grafico_consenso = consenso_total.sort_values("TX_ABSTENCAO")

plt.figure(figsize=(10, 6))

plt.barh(
    grafico_consenso["NM_MUNICIPIO"] + " - " + grafico_consenso["SG_UF"],
    grafico_consenso["TX_ABSTENCAO"]
)

plt.xlabel("Taxa de abstenção (%)")
plt.title("Municípios identificados pelos 3 métodos")

for i, valor in enumerate(grafico_consenso["TX_ABSTENCAO"]):
    plt.text(valor + 0.3, i, f"{valor:.1f}%")

plt.tight_layout()

plt.savefig(
    "outputs/grafico_consenso_3_metodos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nComparação entre os métodos:")

print(
    "IQR + Z-score:",
    (
        municipios["OUTLIER_IQR"]
        & municipios["OUTLIER_ZSCORE"]
    ).sum()
)

print(
    "IQR + Isolation Forest:",
    (
        municipios["OUTLIER_IQR"]
        & municipios["OUTLIER_ISOLATION"]
    ).sum()
)

print(
    "Z-score + Isolation Forest:",
    (
        municipios["OUTLIER_ZSCORE"]
        & municipios["OUTLIER_ISOLATION"]
    ).sum()
)

print(
    "Os 3 métodos:",
    (municipios["TOTAL_METODOS"] == 3).sum()
)

resultado_consenso.to_csv(
    "outputs/municipios_consenso.csv",
    index=False,
    encoding="utf-8-sig"
)

consenso_total.to_csv(
    "outputs/municipios_3_metodos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos de resultados salvos na pasta outputs.")