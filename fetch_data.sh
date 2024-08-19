#!/bin/bash

# Fetch DeepSpeech
echo -e "\nDownloading DeepSpeech..."
wget -P ./ds_graph https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xf ./ds_graph/deepspeech-0.1.0-models.tar.gz -C ./ds_graph
cp -r ./ds_graph/models/* ./ds_graph
echo -e "\nDeepSpeech downloaded."


# Fetch FLAME data
echo -e "\nDownloading FLAME..."

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://flame.is.tue.mpg.de/ and agree to the FLAME license terms."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2019.zip&resume=1' -O './flame/FLAME2019.zip' --no-check-certificate --continue
unzip ./flame/FLAME2019.zip -d ./flame
echo -e "\nFLAME downloaded."


# Fetch VOCA data
echo -e "\nDownloading VOCA..."

echo -e "\nBefore you continue, you must register at https://voca.is.tue.mpg.de/ and agree to the VOCA license terms."
read -p "Username (VOCA):" username
read -p "Password (VOCA):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&resume=1&sfile=model.zip' -O './model/model.zip' --no-check-certificate --continue
unzip ./model/model.zip -d ./model/
echo -e "\nVOCA downloaded."
