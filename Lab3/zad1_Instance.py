#plitki omotač oko podataka. Jednostavni i korisni načini implementacije ovakvog razreda su razredi podataka. Alternativa su imenovane ntorke. Koristim immenovane ntorke.

from collections import namedtuple
import csv

class Instance:
    def __init__(self, csv_file) -> None:
        
        MovieReviews = namedtuple('MovieReviews', 'field, pos_neg')

        self.moviereviews = MovieReviews
        self.csv_file = csv_file

        simple_tuples = []

        for review in map(MovieReviews._make, csv.reader(open(csv_file)) ):
            simple_tuples.append( (review.field.split(), review.pos_neg.strip()) )

        self.reviews = simple_tuples


    def frequencies(self):
        
        #frequencies
        words = {}
        labels = {}

        for review in map(self.moviereviews._make, csv.reader(open(self.csv_file)) ):
            
            for el in review.field.split():
                if el in words:
                    words[el] += 1
                else:
                    words[el] = 1

            if review.pos_neg.strip() in labels:
                labels[review.pos_neg.strip()] += 1
            else:
                labels[review.pos_neg.strip()] = 1


        self.text_frequencies = words
        self.label_frequencies = labels


