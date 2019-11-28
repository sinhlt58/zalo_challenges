from underthesea import sent_tokenize

out_file = open('test_bert_wiki', 'w', encoding='utf-8')

is_bert = True
with open('test_wiki.txt', 'r', encoding='utf-8') as f:
    is_title = True
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            if line.startswith('<doc id='):
                is_title = True
            elif line.startswith('</doc>'):
                # process current document
                pass
            else:
                if is_title:
                    is_title = False
                    continue
                sents = sent_tokenize(line)
                out_file.write('\n'.join(sents))
                out_file.write('\n')

                if is_bert:
                    out_file.write('\n')

out_file.close()
