from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data_akhir.csv', sep=';', usecols=('content_clear','polarity'))
#=====================Positif=========================#
wc_positif = data[data['polarity'] == 'positif']
all_text_pst = ' '.join(word for word in wc_positif['content_clear'])
wordcloud = WordCloud(colormap = 'Blues', width = 1000, mode = 'RGBA', background_color = 'white').generate(all_text_pst)
plt.figure(figsize = (20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.show()

#=====================Negatif=========================#
wc_negatif = data[data['polarity'] == 'negatif']
all_text_pst = ' '.join(word for word in wc_negatif['content_clear'])
wordcloud = WordCloud(colormap = 'Reds', width = 1000, mode = 'RGBA', background_color = 'white').generate(all_text_pst)
plt.figure(figsize = (20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.show()
