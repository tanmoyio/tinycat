from gapencoder import GapEncoder

enc = GapEncoder(n_components=2)
X = [['paris, FR'], ['Paris'], ['London, UK'], ['Paris, France'],
             ['london'], ['London, England'], ['London'], ['Pqris']]
enc.fit(X)
print(enc.get_feature_names_out())
