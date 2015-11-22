from blocks.extensions import SimpleExtension

from pandas import DataFrame
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText 
from email.mime.image import MIMEImage

import smtplib
import ipdb

class Plot(SimpleExtension):
    """ Alternative plot extension for blocks.
    Parameters
    ----------
    document : str
        The name of the plot file. Use a different name for each
        experiment if you are storing your plots.
    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, channels, email=True, **kwargs):

        self.plots = {}
        self.document = document
        self.num_plots = len(channels)
        self.channels = channels
        self.all_channels = list(set([x for small in channels for x in small]))

        self.document=document

        self.strFrom = 'sotelo@iro.umontreal.ca'
        self.strTo = 'rdz.sotelo@gmail.com'
        self.email = email

        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        df = DataFrame.from_dict(log, orient='index')
        df=df.interpolate('index')
        df=df[self.all_channels].astype(float)

        fig, axarr = pyplot.subplots(self.num_plots, sharex=True)

        if self.num_plots > 1:
            for i, channel in enumerate(self.channels):
                df[channel].plot( ax = axarr[i])
        else:
            df[self.channels[0]].plot()

        pyplot.savefig(self.document)
        pyplot.close()

        if self.email:
            msgRoot = MIMEMultipart('related')
            msgRoot['Subject'] = 'Blocks experiment: ' + self.document
            msgRoot['From'] = self.strFrom   
            msgRoot['To'] = self.strTo

            msgAlternative = MIMEMultipart('alternative')
            msgRoot.attach(msgAlternative)

            msgText = MIMEText('Alternative view.')
            msgAlternative.attach(msgText)

            msgText = MIMEText('Results:<br><img src="cid:image"><br> End.', 'html')
            msgAlternative.attach(msgText)

            fp = open(self.document, 'rb')
            msgImage = MIMEImage(fp.read())
            fp.close()

            # Define the image's ID as referenced above
            msgImage.add_header('Content-ID', '<image>')
            msgRoot.attach(msgImage)

            smtp = smtplib.SMTP()
            smtp.connect('localhost')
            smtp.sendmail(self.strFrom, self.strTo, msgRoot.as_string())
            smtp.quit()



