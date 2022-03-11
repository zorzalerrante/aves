#!/usr/bin/env bash

set -e

cd "data/external"

mkdir casen_2020
cd casen_2020

wget -nc http://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2020/Libro_de_codigos_Base_de_Datos_Casen_en_Pandemia_2020.pdf
wget -nc http://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2020/Casen_en_Pandemia_2020_STATA.dta.zip
unzip Casen_en_Pandemia_2020_STATA.dta.zip

cd ../../..