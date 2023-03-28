#!/usr/bin/env bash

set -e

cd "data/external"
mkdir OSM
cd OSM

wget -nc https://download.geofabrik.de/south-america/chile-latest.osm.pbf

cd "../../.."