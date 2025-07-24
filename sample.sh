CORPUS_RELEASE="CC-MAIN-2023-23"

for group in $(ls data/${CORPUS_RELEASE}); do
  file_count=$(find data/${CORPUS_RELEASE}/${group} -type f | wc -l)
  # echo "file_count: ${file_count} for group: ${group}"
  python src/sample.py \
    --corpus_release ${CORPUS_RELEASE} \
    --group ${group} \
    --file_count ${file_count}
  break
done

# poetry run python src/setfit-predict.py