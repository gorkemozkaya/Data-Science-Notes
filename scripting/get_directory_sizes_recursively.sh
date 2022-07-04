for i in G M K; do du -ah [folder name] | grep [0-9]$i | sort -nr -k 1; done | head -n 11

