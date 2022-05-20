#!/usr/bin/env bash

set -e

cd "data/external"

[ -d wiki2vec ] || mkdir wiki2vec
cd wiki2vec

wget -nc http://wikipedia2vec.s3.amazonaws.com/models/es/2018-04-20/eswiki_20180420_100d.txt.bz2

cd ../../..