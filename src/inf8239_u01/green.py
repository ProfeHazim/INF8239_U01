def pareto_flags(df, score="f1_macro", cost="fit_median_s"):
    """Calcula las banderas de optimalidad de Pareto para un DataFrame de modelos."""
    flags = []
    for _, row in df.iterrows():
        dominated = ((df[score] >= row[score]) & (df[cost] <= row[cost]) &((df[score] > row[score]) | (df[cost] < row[cost]))).any()
        flags.append(not bool(dominated))
    return flags