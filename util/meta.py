label_list = ['ARTS',
              'ARTS & CULTURE',
              'BLACK VOICES',
              'BUSINESS',
              'COLLEGE',
              'COMEDY',
              'CRIME',
              'CULTURE & ARTS',
              'DIVORCE',
              'EDUCATION',
              'ENTERTAINMENT',
              'ENVIRONMENT',
              'FIFTY',
              'FOOD & DRINK',
              'GOOD NEWS',
              'GREEN',
              'HEALTHY LIVING',
              'HOME & LIVING',
              'IMPACT',
              'LATINO VOICES',
              'MEDIA',
              'MONEY',
              'PARENTING',
              'PARENTS',
              'POLITICS',
              'QUEER VOICES',
              'RELIGION',
              'SCIENCE',
              'SPORTS',
              'STYLE',
              'STYLE & BEAUTY',
              'TASTE',
              'TECH',
              'TRAVEL',
              'WEDDINGS',
              'WEIRD NEWS',
              'WELLNESS',
              'WOMEN',
              'WORLD NEWS',
              'WORLDPOST']

label_list2 = ['ARTS', 'ARTS & CULTURE', 'BLACK VOICES', 'BUSINESS', 'COLLEGE', 'COMEDY', 'CRIME', 'CULTURE & ARTS',
              'DIVORCE', 'EDUCATION', 'ENTERTAINMENT', 'ENVIRONMENT', 'FIFTY', 'FOOD & DRINK', 'GOOD NEWS', 'GREEN',
              'HEALTHY LIVING', 'HOME & LIVING', 'IMPACT', 'LATINO VOICES', 'MEDIA', 'MONEY', 'PARENTING', 'PARENTS',
              'POLITICS', 'QUEER VOICES', 'RELIGION', 'SCIENCE', 'SPORTS', 'STYLE', 'STYLE & BEAUTY', 'TASTE', 'TECH',
              'TRAVEL', 'WEDDINGS', 'WEIRD NEWS', 'WELLNESS', 'WOMEN', 'WORLD NEWS', 'WORLDPOST']

side_list = ['least', 'right', 'right-center', 'left-center', 'left']

if __name__ == '__main__':
    for i in range(label_list.__len__()):
        if label_list[i] != label_list2[i] or label_list2.__len__() != label_list.__len__():
            print('hehe')
