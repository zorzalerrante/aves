#!/usr/bin/env bash

set -e

cd "data/external"

mkdir plebiscito_2020
cd plebiscito_2020

wget -nc https://oficial.servel.cl/wp-content/uploads/2021/08/Resultados_Plebiscito_Constitucion_Politica_2020.zip
unzip Resultados_Plebiscito_Constitucion_Politica_2020.zip

cd ../../..