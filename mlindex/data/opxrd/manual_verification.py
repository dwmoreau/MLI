import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def interactive_peak_plot(data):
    """
    Plot I vs q with clickable peak position vertical lines.
    
    Parameters
    ----------
    data : dict
        Dictionary with keys 'q', 'I', and 'peak_positions'.
        
    Returns
    -------
    tuple (dict, bool)
        Updated dictionary with remaining peak positions after figure is closed,
        and a boolean indicating whether the entry was accepted (True) or rejected (False).
    """
    rejected = [False]
    held_key = [None]

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(data['q'], data['I'], color='blue', label='I vs q')
    ax.set_xlabel('q')
    ax.set_ylabel('I')
    ax.set_xlim([data['q'][0], data['peak_positions'][-1] + 0.02])
    ax.set_title('Click peak to remove | Hold "a" + click to add | Press "r" to reject | Close to accept')

    peak_lines = []
    for pos in data['peak_positions']:
        line = ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, picker=5)
        peak_lines.append(line)

    ax.legend()

    def on_pick(event):
        if held_key[0] is not None:
            return
        line = event.artist
        if line in peak_lines:
            x_val = line.get_xdata()[0]
            line.remove()
            peak_lines.remove(line)
            if x_val in data['peak_positions']:
                data['peak_positions'].remove(x_val)
            fig.canvas.draw_idle()

    def on_click(event):
        if held_key[0] == 'a' and event.inaxes == ax:
            x_val = event.xdata
            line = ax.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, picker=5)
            peak_lines.append(line)
            data['peak_positions'].append(x_val)
            data['peak_positions'].sort()
            fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == 'r':
            rejected[0] = True
            ax.set_title('REJECTED — closing...')
            fig.canvas.draw_idle()
            plt.close(fig)
        elif event.key == 'a':
            held_key[0] = 'a'

    def on_key_release(event):
        if event.key == 'a':
            held_key[0] = None

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    plt.show()

    return data, not rejected[0]


def main():
    input_file = 'CNRS_output_data.json'
    output_file = 'CNRS_output_data_verified_0.json'

    df = pd.read_json(input_file)

    start_index = int(sys.argv[1])
    verified_records = []

    for index in range(start_index, len(df)):
        print(f"\n--- Entry {index}/{len(df)} ---")
        entry = df.loc[index].to_dict()
        updated_entry, accepted = interactive_peak_plot(entry)

        if accepted:
            verified_records.append(updated_entry)
            print(f"Entry {index} accepted.")
        else:
            print(f"Entry {index} rejected.")

        verified_df = pd.DataFrame(verified_records)
        verified_df.to_json(output_file, orient='index', indent=2)
        print(f"Checkpoint saved ({len(verified_records)} accepted).")

    print(f"\nDone. {len(verified_records)} entries accepted out of {len(df)} total.")


if __name__ == '__main__':
    main()