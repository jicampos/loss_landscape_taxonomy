# Use the official Miniconda image as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /

# Clone the GitHub repository into the container
RUN git clone https://github.com/balditommaso/loss_landscape_taxonomy.git .


# Copy the Conda environment definition file (environment.yml) into the container
# COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "loss_landscape", "/bin/bash", "-c"]

# Activate the Conda environment
RUN echo "source activate loss_landscape" >> ~/.bashrc
ENV PATH /opt/conda/envs/loss_landscape/bin:$PATH

# Specify the command to run on container start
CMD ["conda", "run", "--no-capture-output", "-n", "loss_landscape", "/bin.bash"]
