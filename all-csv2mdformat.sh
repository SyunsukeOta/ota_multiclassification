base_path=outputs/ruri-v3-310m/2025-06-30
setfit_path=02-
out_dir=analysis_results/setfit_hypothesis_mdtable
find $base_path -maxdepth 1 -type d |\
grep "${setfit_path}[0-9]\{2\}-[0-9]\{2\}\.[0-9]\{6\}" |\
sort |\
while read -r path; do
  prefix=$(echo "$path" | grep -o '[0-9]\{6\}$')
  echo "prefix: $prefix"
  python src/data_analyze/csv_to_markdown.py --csv_path "$path/log.csv"
  # python src/data_analyze/csv_to_markdown.py --csv_path "$path/log.csv" --out_path "${out_dir}/${prefix}.txt"
done

# outputs/ruri-v3-310m/2025-06-29/09-42-40.291126/log.csv
# outputs/ruri-v3-310m/2025-06-29/09-42-40.309898/log.csv
# outputs/ruri-v3-310m/2025-06-29/09-42-41.609630/log.csv
# outputs/ruri-v3-310m/2025-06-29/09-42-41.623052/log.csv
# outputs/ruri-v3-310m/2025-06-29/09-42-41.630394/log.csv
# outputs/ruri-v3-310m/2025-06-29/09-42-41.644306/log.csv
