# Pull thermoengine pkg from docker container
# FROM registry.gitlab.com/enki-portal/thermoengine:master
FROM registry.gitlab.com/enki-portal/thermoengine:staging

# Copy local app to container
COPY . ${HOME}/app
COPY hekla.csv ${HOME}/hekla.csv
COPY FracCrystExercise.ipynb ${HOME}/FracCrystExercise.ipynb

# Reset permissions to allow thermoengine to run
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Enable settings for cloud apps
# USER root
# RUN pip install --no-cache-dir appmode
# RUN jupyter nbextension enable --py --sys-prefix appmode
# RUN jupyter serverextension enable --py --sys-prefix appmode
# USER ${NB_USER}

# Install app inside container
WORKDIR ${HOME}/app
RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install mpltern
RUN pip install pyrolite
# RUN make devinstall
WORKDIR ${HOME}