import matplotlib
matplotlib.use('agg')
from batchgenerators.utilities.file_and_folder_operations import join
import seaborn as sns
import matplotlib.pyplot as plt

class nnUNetLogger(object):
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {}  # Changed: Now fully dynamic
        self.verbose = verbose

    def log(self, key, value, epoch: int):
        """Modified: Now accepts any key dynamically."""
        if key not in self.my_fantastic_logging:
            self.my_fantastic_logging[key] = []
        
        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            self.my_fantastic_logging[key][epoch] = value

        # Original EMA dice handling (preserved)
        if key == 'mean_fg_dice':
            new_ema = value if not self.my_fantastic_logging.get('ema_fg_dice') else \
                     self.my_fantastic_logging['ema_fg_dice'][-1] * 0.9 + value * 0.1
            self.log('ema_fg_dice', new_ema, epoch)

    def plot_progress_png(self, output_folder):
        """Original function with added dynamic key support."""
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        
        # Plot 1: Original loss + dice (unchanged)
        ax = ax_all[0]
        ax2 = ax.twinx()
        x = list(range(epoch + 1))
        ax.plot(x, self.my_fantastic_logging.get('train_losses', [0]*len(x))[:epoch + 1], 'b-', label="loss_tr", linewidth=4)
        ax.plot(x, self.my_fantastic_logging.get('val_losses', [0]*len(x))[:epoch + 1], 'r-', label="loss_val", linewidth=4)
        ax2.plot(x, self.my_fantastic_logging.get('mean_fg_dice', [0]*len(x))[:epoch + 1], 'g:', label="pseudo dice", linewidth=3)
        ax2.plot(x, self.my_fantastic_logging.get('ema_fg_dice', [0]*len(x))[:epoch + 1], 'g-', label="pseudo dice (mov. avg.)", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # Plot 2: New uncertainty metrics (added)
        if 'train_error_rate' in self.my_fantastic_logging:
            ax = ax_all[1]
            ax.plot(x, self.my_fantastic_logging['train_error_rate'][:epoch + 1], 'm-', label="train_error", linewidth=4)
            ax.plot(x, self.my_fantastic_logging.get('val_error_rate', [0]*len(x))[:epoch + 1], 'm--', label="val_error", linewidth=4)
            ax.set_xlabel("epoch")
            ax.set_ylabel("error rate")
            ax.legend(loc=(0, 1))

        # Plot 3: Original epoch timing (unchanged)
        ax = ax_all[2]
        ax.plot(x, [i-j for i,j in zip(
            self.my_fantastic_logging.get('epoch_end_timestamps', [0]*len(x)),
            self.my_fantastic_logging.get('epoch_start_timestamps', [0]*len(x)))][:epoch + 1], 
            'b-', label="epoch duration", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        plt.tight_layout()
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        """Original unchanged."""
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        """Original unchanged."""
        self.my_fantastic_logging = checkpoint