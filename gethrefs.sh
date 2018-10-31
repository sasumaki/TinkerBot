#!/bin/bash

echo $1
curl -L $1 -o site.txt
grep -Po '(?<=href="[^https])[^"]*' site.txt > base.txt
shuf base.txt > rhrefs.txt
rm base.txt
i=0
prefix=$1
while read link; do
 i=$((i + 1))
 if grep -Fxq "$link" rhrefs.txt
 then
  if [[ $link == *"$prefix"* ]]
  then 
  link = "$(echo "$link" | grep -oP "^$prefix\K.*")"
  fi
  curl -L -s $1/$link -o site.txt
  grep -Po '(?<=href="[^https])[^"]*' site.txt >> rhrefs.txt
  if [ $i == 1000 ]
  then
     sort < rhrefs.txt | uniq > hrefs.txt
     exit
  fi
 fi
done <rhrefs.txt
rm hrefs.txt
mv rhrefs.txt hrefs.txt
rm site.txt
