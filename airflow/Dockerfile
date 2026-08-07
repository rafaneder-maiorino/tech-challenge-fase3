# Imagem do Airflow com as dependências de treino do projeto.
#
# O código de src/ é montado como volume (não copiado), então o build só existe
# para instalar bibliotecas. Rebuild necessário apenas quando estas versões
# mudarem — mexer nas DAGs ou em src/ não exige rebuild.
FROM apache/airflow:2.11.0-python3.11

# ATENÇÃO — estas versões DEVEM ser as mesmas do uv.lock do projeto.
#
# O motivo é a compatibilidade do artefato: o modelo treinado aqui é
# desserializado pela API, que instala as versões do uv.lock. joblib/pickle de
# um Pipeline scikit-learn não é compatível entre versões diferentes do
# scikit-learn — treinar na 1.6.1 (a que vem na imagem do Airflow) e carregar na
# 1.9.0 (a da API) produz InconsistentVersionWarning e risco de erro silencioso.
#
# Para conferir se ainda estão sincronizadas:
#   uv run python -c "import tomllib;d=tomllib.load(open('uv.lock','rb'));\
#     print({p['name']:p['version'] for p in d['package'] \
#     if p['name'] in {'pandas','scikit-learn','huggingface-hub','pyarrow','joblib'}})"
RUN pip install --no-cache-dir \
        "pandas==3.0.5" \
        "scikit-learn==1.9.0" \
        "huggingface-hub==1.24.0" \
        "pyarrow==25.0.0" \
        "joblib==1.5.3"

# numpy fica em 1.26.4 (o que a imagem já traz) — pandas 3.0.5 e scikit-learn
# 1.9.0 são compatíveis com ele, o que evita a quebra de ABI que um upgrade
# para numpy 2.x causaria nos pacotes compilados do Airflow.
#
# `pip check` reclama de apache-airflow-providers-google e -snowflake, que
# pedem pandas<2.2. Ambos são providers não usados por este projeto (as DAGs
# usam só PythonOperator) e não são importados em runtime. Ver docs.
