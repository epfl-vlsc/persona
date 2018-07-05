from subprocess import call
import json
import plotly as py
import plotly.graph_objs as go


def main():

    threads = [1, 2, 4, 8, 16, 32, 40, 47, 48]
    # threads = []
    threads.reverse()
    times_per_threads = []
    for t in threads:
        args = [
            "./persona",
            "protein_cluster",
            "-c",
            "modules/protein_cluster/protclstr.json",
            "-n",
            "1",
            "-t",
            str(t),
            # "--do-allall",
            "-f",
            "/scratch/proteomes/agd/bacteria_chlorobiaceae/bigchunk/chll2/chll2_metadata.json",
            "/scratch/proteomes/agd/bacteria_chlorobiaceae/bigchunk/chlch/chlch_metadata.json",
            "/scratch/proteomes/agd/bacteria_chlorobiaceae/bigchunk/chll7/chll7_metadata.json",
            "/scratch/proteomes/agd/bacteria_chlorobiaceae/bigchunk/chlp8/chlp8_metadata.json",
        ]

        call(args)

        with open("runtime.txt") as infile:
            line = infile.readline()
            times_per_threads.append(1.0 / float(line))  # time per work (thruput)

    print("times is {}".format(times_per_threads))
    # create graph
    # threads = [1, 2, 4, 8, 16, 32, 40, 47, 48]
    # times_per_threads = [10, 20, 40, 80, 160, 320, 400, 470, 480]

    py.offline.plot(
        {
            "data": [go.Scatter(x=threads, y=times_per_threads)],
            "layout": go.Layout(
                title="Thread Scaling (Clustering only)",
                xaxis=dict(
                    title="# Threads",
                    # titlefont=dict(
                    # family="Courier New, monospace", size=18, color="#7f7f7f"
                    # ),
                ),
                yaxis=dict(
                    title="Throughput (1/s)",
                    # titlefont=dict(
                    # family="Courier New, monospace", size=18, color="#7f7f7f"
                    # ),
                ),
            ),
        },
        image="png",
        auto_open=False,
    )

    with open("thread_scaling.json", "w") as outfile:
        json.dump(times_per_threads, outfile)


if __name__ == "__main__":
    main()
