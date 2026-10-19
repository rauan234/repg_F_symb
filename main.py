from read_write_GPT import read_from_file
from load_rep_data import load_repg, load_subgroup


def main():
    data = read_from_file("D8_data.json")

    # load untwisted rep data for G
    repg = load_repg(data["G"])

    # compute Rep(G) F-symbols and check unitarity and pentagon equation
    repg.compute_F_symb()
    repg.check_F_unitary()
    repg.check_pentagon_eqs()
    print()

    # load data about twisted reps of subgroups
    for sg_data in data["subgroups"]:  # iterate over subgroups H
        # list of a RepSubgroupTwisted object for each cohomology class for H
        reph_list = load_subgroup(repg, sg_data)

        for reph in reph_list:
            reph.compute_F_right()
            reph.compute_F_left()
            reph.compute_F_lr()

            #repg.print_F_symb()
            #reph.print_F_right()

            reph.check_F_unitary()
            reph.check_pentagon_eqs()

        print()


if __name__ == "__main__":
    main()
