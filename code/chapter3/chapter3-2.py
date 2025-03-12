from  matplotlib import pyplot as plt
from collections import Counter

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Stroy"]
num_oscars = [5, 11, 3, 8, 10]
plt.figure()
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")
plt.ylabel("# of Academy Awards")

plt.xticks(range(len(movies)), movies)

plt.savefig("/app/picture/chap3-2.png")

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

histgram = Counter(min(grade // 10 * 10, 90) for grade in grades)
plt.figure()
plt.bar([x + 5 for x in histgram.keys()],
        histgram.values(),
        10,
        edgecolor=(0, 0, 0))
plt.axis([-5, 105, 0, 5])

plt.xticks([10 * i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("# of Studensts")
plt.title("Distribution of Exam 1 Grades")
plt.savefig("/app/picture/chap3-3.png")

mentions = [500, 505]
years = [2017, 2018]

plt.figure()
plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

plt.ticklabel_format(useOffset=False)

plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Look at the 'Huge' Increase!")

plt.savefig("/app/picture/chap3-4.png")

plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore")
plt.savefig("/app/picture/chap3-5.png")