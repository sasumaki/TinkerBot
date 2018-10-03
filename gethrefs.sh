#!/bin/bash

curl -L https://www.varusteleka.com -o site.txt
grep -Po '(?<=href="[^https])[^"]*' site.txt > rhrefs.txt
i=0
while read link; do
 curl -L https://varusteleka.com/$link -o site.txt
 grep -Po '(?<=href="[^https])[^"]*' site.txt >> rhrefs.txt
 i=$((i + 1))
 if [ $i == 200 ]
 then
    sort < rhrefs.txt | uniq > hrefs.txt
    exit
 fi
done <rhrefs.txt
rm hrefs.txt
mv rhrefs.txt hrefs.txt
rm site.txt
