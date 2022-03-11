#!/usr/bin/env bash

set -e

cd "data/external"

mkdir casen_2017
cd casen_2017

wget -nc http://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2017/Libro_de_Codigos_Casen_2017.pdf
wget -nc http://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2017/casen_2017_stata.rar
unrar casen_2017_stata.rar

cd ../../..