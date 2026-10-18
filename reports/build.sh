#!/bin/bash
# Build both reports. Requires tectonic (https://tectonic-typesetting.github.io/).
set -e
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-tectonic}
"$TECTONIC" -X compile report_project.tex
"$TECTONIC" -X compile report_architecture.tex
ls -la ./*.pdf
