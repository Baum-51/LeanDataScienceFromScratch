import enum, random

# 列挙値の型付き集合であるEnumを使うことで
# コードがより説明的で読みやすくなる
class Kid(enum.Enum):
    BOY  = 0
    GIRL = 1
    
def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(２人とも女の子 | １人目が女の子)　　　:", both_girls / older_girl)  # 0.514 ~ 1/2
print("P(２人とも女の子 | どちらか１人が女の子):", both_girls / either_girl) # 0.342 ~ 1/3

# どちらか片方が女の子である条件の下での２人とも女の子である確率
# どちらか片方が女の子である確率は3/4、２人とも女の子である確率は1/4