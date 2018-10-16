#!/bin/bash

curl -L https://www.varusteleka.com -o site.txt
grep -Po '(?<=href="[^https])[^"]*' site.txt > base.txt
shuf base.txt > rhrefs.txt
rm base.txt
i=0
prefix="https://varusteleka.com/"
while read link; do
 i=$((i + 1))
 if grep -Fxq "$link" rhrefs.txt
 then
 echo $link
 echo $prefix
  if [[ $link == *"$prefix"* ]]
  then 
  link = "$(echo "$link" | grep -oP "^$prefix\K.*")"
  fi
  echo $asd
  echo https://varusteleka.com/$link $i
  curl -L -s https://varusteleka.com/$link -o site.txt
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
