#!/usr/bin/env bash

set -e

cd "data/external"

[ -d plebiscito_2020 ] || mkdir plebiscito_2020
cd plebiscito_2020

wget -nc https://oficial.servel.cl/wp-content/uploads/2021/08/Resultados_Plebiscito_Constitucion_Politica_2020.zip
unzip -o Resultados_Plebiscito_Constitucion_Politica_2020.zip

cd ../../..