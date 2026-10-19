# repg_F_symb
Compute the fusion category Rep(G), as well as its module categories and module functors, where G is a finite group. This code is to be used in conjunction with AI, as explained in readme. The code is hand-written by me, except for explicitly marked pieces whose output is checked by my code.

HOW TO USE
1. Select a finite group G.
2. In prompt.txt, replace "[INSERT GROUP DESCRIPTION. example: "Let G be S3, the group of permutations of 3 elements"]" with a description of your group G.
3. Run the modified prompt in an LLM of your choice.
4. Read off LLM output as read_from_file("S3_data.json"), as illustrated in main.py. Then, as in main.py, run load_repg(data["G"]) and related commands to convert json data into SymPy-based classes. This not only loads the data but also CHECKS IT FOR VALIDITY. This way, even if the LLM hallucinated, any mathematical inconsistency is caught.
5. Access F-symbols and any other requested information.
