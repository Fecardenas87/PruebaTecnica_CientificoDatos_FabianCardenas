"""
ETL_EVOLUCION

Objetivo:
Construir un dataset enriquecido a partir de las fuentes EVOLUCION, PAGOS y TELEFONOS,
aplicando procesos de limpieza, estandarización, validación de llaves y generación
de variables de negocio para análisis y modelado posterior.

Flujo general:
1. Carga de fuentes crudas
2. Limpieza y normalización de columnas
3. Agregación de pagos aprobados
4. Construcción de llaves técnicas para cruces consistentes
5. Enriquecimiento con información de teléfonos
6. Generación de variables de negocio a nivel producto y cliente
7. Exportación del dataset final

Salida:
- evolucion_enriquecida.csv
- df_evolucion_enriquecida.txt
"""


import pandas as pd
from pathlib import Path
import unicodedata
import re

"RUTAS -- Definen la ubicación base y los directorios de entrada/salida"

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_RAW = BASE_DIR / "data" / "Datos_SinProcesar"
DATA_PROCESSED = BASE_DIR / "data" / "Datos_Procesados"


"FUNCIONES DE CARGA -- Lectura de archivos fuente sin transformación"
def cargar_evolucion():
    
    df = pd.read_csv(
        DATA_RAW / "EVOLUCION.txt",
        sep="\t",
        # encoding="utf-8"
        encoding="latin1"
    )
    return df

def cargar_pagos():
    
    df = pd.read_csv(
        DATA_RAW / "PAGOS.txt",
        sep="|",
        encoding="latin1"
    )
    return df

def cargar_telefonos():
    
    df = pd.read_csv(
        DATA_RAW / "TELEFONOS.txt",
        sep="\t",
        encoding="latin1"
    )
    return df


"FUNCIONES DE LIMPIEZA -- Normalización de nombres de columnas y variables numéricas"
# Normalizan nombres de columnas, minúsculas, quita espacios y usa _
def limpiar_columnas(df):
    
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

# Limpio columnas monetarias númericasd, elimino símbolos y convierto a numerico
def limpiar_columnas_numericas(df, columnas):
    
    for col in columnas:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def extraer_numero(texto):
    if pd.isna(texto):
        return None

    match = re.search(r"\d+", str(texto))
    return match.group(0) if match else None

def normalizar_texto(texto):
    
    if pd.isna(texto):
        return None

    texto = str(texto)

    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = texto.lower()

    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

# Limpieza de telefonos
def limpiar_telefono(valor):
    
    if pd.isna(valor):
        return None

    valor = str(valor)
    valor = re.sub(r"\D", "", valor)  # solo números

    return valor if valor != "" else None

#Celulares 10 digitos empiezan por 7, segundo digito entre 0 y 5
def es_celular_valido(numero):
   
    #if numero is None:
    if not isinstance(numero, str):
        return False
    return bool(re.match(r"^7[0-5][0-9]{8}$", numero))
    

#Teléfonos 7 digitos empiezan entre 1 y 5
def es_fijo_valido(numero):
   
    #if numero is None:
    if not isinstance(numero, str):
        return False

    return bool(re.match(r"^[1-5][0-9]{6}$", numero))

#Clasifico numeros como CELULAR o FIJO -- None si no es ninguno de los anteriores
def clasificar_telefono(numero):
   
    if es_celular_valido(numero):
        return "CELULAR"
    if es_fijo_valido(numero):
        return "FIJO"
    return None

#Acá depuro telefonos!!!
def depurar_telefonos(df):
   
    df = df.copy()
    df = limpiar_columnas(df)
    df.rename(columns={"telefono_1": "telefono"}, inplace=True)

    df["telefono_limpio"] = df["telefono"].apply(limpiar_telefono)
    df["tipo_telefono"] = df["telefono_limpio"].apply(clasificar_telefono)

    # Me quedo solo con teléfonos válidos
    df = df[df["tipo_telefono"].notna()]

    return df


#FORMATO ANCHO - UNA COLUMNA POR TELEFONO
def telefonos_formato_ancho(df):
    
    df = df.copy()
    orden_tipo = {"CELULAR": 0, "FIJO": 1}
    df["orden"] = df["tipo_telefono"].map(orden_tipo)

    df = df.sort_values(["identificacion", "orden", "telefono_limpio"])
    df["n"] = df.groupby(["identificacion", "tipo_telefono"]).cumcount() + 1
    df["columna"] = (
        df["tipo_telefono"].str.lower()
        + "_"
        + df["n"].astype(str)
    )

    # Pivot
    df_ancho = (
        df.pivot(
            index="identificacion",
            columns="columna",
            values="telefono_limpio"
        )
        .reset_index()
    )

    return df_ancho


# Estandarizo y homologo los productos
def estandarizar_producto(valor):
    
    texto = normalizar_texto(valor)

    if texto is None:
        return None

    if "credito" in texto or "cx" in texto:
        return "CREDITO"
    if "tarjeta" in texto:
        return "TARJETA"
    if "hipotec" in texto:
        return "HIPOTECARIO"
    if "vehicul" in texto:
        return "VEHICULO"
    if "rotativo" in texto:
        return "ROTATIVO"

    return "OTRO"


"AGREGACIÓN DE PAGOS"
#Agrego pagos aprobados por cuenta

def agregar_pagos(df_pagos):

    # print("\n--- VALIDACIÓN ESTADO_PAGO ---")
    # print(df_pagos["estado_pago"].value_counts(dropna=False))
    # print("\n--- HEAD PAGOS ---")
    # print(df_pagos.head(5))

    pagos_agg = (
        df_pagos
        .query("estado_pago == 'APROBADO'")
        .groupby("cuenta", as_index=False)
        .agg(
            total_pagos=("pagos", "sum"),
            num_pagos=("pagos", "count")
        )
    )

    # print("\n--- PAGOS AGREGADOS (POST FILTRO) ---")
    # print(pagos_agg.head())
    # print("Registros agregados:", pagos_agg.shape[0])

    return pagos_agg


"EJECUCIÓN"

if __name__ == "__main__":
    df_evolucion = cargar_evolucion()
    df_pagos = cargar_pagos()


    df_evolucion = limpiar_columnas(df_evolucion)
    df_pagos = limpiar_columnas(df_pagos)


    df_evolucion = limpiar_columnas_numericas(
        df_evolucion,
        ["saldo_capital_mes", "pago_minimo", "dias_mora"],
    )

    df_pagos = limpiar_columnas_numericas(
        df_pagos,
        ["pagos"]
    )
    
    df_pagos_agg = agregar_pagos(df_pagos)

    "LLAVES TÉCNICAS"
     # Extraje la parte numérica de la obligación para asegurar cruces consistentes con la tabla de pagos
    df_evolucion["obligacion_num"] = (
        df_evolucion["obligacion"].apply(extraer_numero)
    )

    df_pagos_agg["cuenta_num"] = (
        df_pagos_agg["cuenta"].astype(str)
    )


    #DEJO EL MISMO DATATYPE PARA EL MERGE
    df_evolucion["obligacion"] = df_evolucion["obligacion"].astype(str)
    df_pagos_agg["cuenta"] = df_pagos_agg["cuenta"].astype(str)

    # print("\n--- VALIDACIÓN LLAVES ---")
    # print(df_evolucion[["obligacion", "obligacion_num"]].head(10))
    # print(df_pagos_agg[["cuenta", "cuenta_num"]].head(10))

    # print("\nTipos:")
    # print("obligacion:", df_evolucion["obligacion"].dtype)
    # print("cuenta:", df_pagos_agg["cuenta"].dtype)


    "CRUZO EVOL CON PAGOS -- LEFT JOIN"
    df_evolucion_enriquecida = df_evolucion.merge(
        df_pagos_agg,
        # left_on="obligacion",
        # right_on="cuenta",
        left_on="obligacion_num",
        right_on="cuenta_num",
        how="left"
    )

    #df_evolucion_enriquecida.drop(columns=["cuenta"], inplace=True)
    df_evolucion_enriquecida.drop(columns=["cuenta","cuenta_num"], inplace=True)

    print("\n--- VALIDACIÓN POST CRUCE ---")
    print(
        "Registros con pagos:",
        (df_evolucion_enriquecida["num_pagos"] > 0).sum()
    )

    print(
        df_evolucion_enriquecida[
            df_evolucion_enriquecida["num_pagos"] > 0
        ][["identificacion", "obligacion", "num_pagos", "total_pagos"]].head()
    )

    #RELLENO DE NaN    
    df_evolucion_enriquecida[["total_pagos", "num_pagos"]] = (
        df_evolucion_enriquecida[["total_pagos", "num_pagos"]]
        .fillna(0)
    )

    df_evolucion_enriquecida["num_pagos"] = (
    df_evolucion_enriquecida["num_pagos"].astype(int)
    )

    "TELEFONOS"
    df_telefonos = cargar_telefonos()
    #print("\nColumnas TELEFONOS:")
    #print(df_telefonos.columns)
    df_telefonos = depurar_telefonos(df_telefonos)
    df_telefonos_ancho = telefonos_formato_ancho(df_telefonos)

    "CRUZO TELEFONOS CON EVOLUCION -- LEFT JOIN"
    df_evolucion_enriquecida = df_evolucion_enriquecida.merge(
    df_telefonos_ancho,
    on="identificacion",
    how="left"
)

    "COLUMNAS ADICIONALES"
    # TIPO_CLIENTE
    conteo_productos = (
        df_evolucion_enriquecida
        .groupby("identificacion")["obligacion"]
        .nunique()
    )

    df_evolucion_enriquecida["tipo_cliente"] = (
        df_evolucion_enriquecida["identificacion"]
        .map(conteo_productos)
        .apply(lambda x: "MONOPRODUCTO" if x == 1 else "MULTIPRODUCTO")
)

    # ESTADO_ORIGEN
    df_evolucion_enriquecida["estado_origen"] = (
        df_evolucion_enriquecida["estado_cliente"]
        .apply(lambda x: "CON_ACUERDO" if x == 1 else "SIN_ACUERDO")
)

    # SALDO_TOTAL_CLIENTE
    saldo_total = (
        df_evolucion_enriquecida
        .groupby("identificacion")["saldo_capital_mes"]
        .sum()
)

    df_evolucion_enriquecida["saldo_total_cliente"] = (
        df_evolucion_enriquecida["identificacion"]
        .map(saldo_total)
)
    # RANGO_DIAS_DE_MORA
    def rango_dias_mora(dias):
        if dias == 0:
         return "AL DIA"
        elif dias < 30:
            return "MENOS 30"
        elif dias < 60:
            return "MENOS 60"
        elif dias < 90:
            return "MENOS 90"
        elif dias < 120:
            return "MENOS 120"
        elif dias < 180:
            return "MENOS 180"
        elif dias < 360:
            return "MENOS 360"
        elif dias < 540:
            return "MENOS 540"
        else:
            return "MAS DE 540"

    df_evolucion_enriquecida["rango_dias_mora"] = (
        df_evolucion_enriquecida["dias_mora"]
        .apply(rango_dias_mora)
)

    # RANGO_MORA_CLIENTE
    max_mora = (
        df_evolucion_enriquecida
        .groupby("identificacion")["dias_mora"]
        .max()
)

    df_evolucion_enriquecida["rango_mora_cliente"] = (
        df_evolucion_enriquecida["identificacion"]
        .map(max_mora)
        .apply(rango_dias_mora)
)

    # CUMPLE_PAGO_MINIMO
    def cumple_pago_minimo(row):
        if row["pago_minimo"] == 0:
            return "NO CUMPLE"

        ratio = row["total_pagos"] / row["pago_minimo"]

        if ratio > 1:
            return "CUMPLE TOTAL"
        elif ratio >= 0.7:
            return "CUMPLE PARCIAL"
        else:
            return "NO CUMPLE"
        
    df_evolucion_enriquecida["cumple_pago_minimo"] = (
    df_evolucion_enriquecida.apply(cumple_pago_minimo, axis=1)
)



    print("\n--- COLUMNAS DE NEGOCIO ---")
    print(
        df_evolucion_enriquecida[
        [
            "tipo_cliente",
            "estado_origen",
            "saldo_total_cliente",
            "rango_dias_mora",
            "rango_mora_cliente",
            "cumple_pago_minimo"
        ]
    ].head()
)



"GUARDO EL DATASET"
OUTPUT_PATH = DATA_PROCESSED / "evolucion_enriquecida.csv"

df_evolucion_enriquecida.to_csv(
     OUTPUT_PATH,
     index=False,
     encoding="utf-8"
 )

print(f"\nDataset guardado en: {OUTPUT_PATH}")


"EXPORT FINAL A TXT"
OUTPUT_PATH = DATA_PROCESSED / "df_evolucion_enriquecida.txt"

df_evolucion_enriquecida.to_csv(
    OUTPUT_PATH,
    sep="|",
    index=False,
    encoding="utf-8"
)

print(f"\n Archivo final exportado en: {OUTPUT_PATH}")

