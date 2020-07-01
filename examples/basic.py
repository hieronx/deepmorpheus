import deepmorpheus

output = deepmorpheus.tag_from_file("test_input.txt", "ancient-greek")

print()
for sentence in output:
    for word in sentence:
        print(str(word))
    print()
