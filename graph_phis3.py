import pandas as pd
import ROOT
import numpy as np
import os

# --- Configuration ---
input_filename = 'avg2.csv'
output_canvas_name = 'avg_phidiff3a.png'
n_bins = 320
n = n_bins
minn = -0.02
maxx = 3.16
"""
Reads a 6-column CSV, creates a 1D histogram for each column,
and saves them to a single multi-page PDF file using PyROOT.
"""
if not os.path.exists(input_filename):
    print(f"ERROR: Input file not found: {input_filename}")

# 1. READ DATA with Pandas (Efficient and fast)
try:
    # Assuming the CSV has a header row
    df = pd.read_csv(input_filename)
    
except pd.errors.EmptyDataError:
    print("ERROR: CSV file is empty or has no columns.")
except Exception as e:
    print(f"An error occurred reading the CSV: {e}")

# 2. SETUP ROOT CANVAS and FILE

# Create a canvas for plotting
canvas = ROOT.TCanvas("canvas", "Column Histograms", 1920, 1080)
canvas.SetGrid()

# Open the output PDF file (will be multi-page)
#canvas.Print(f"{output_canvas_name}[") # The '[' starts the multi-page PDF

# 3. CREATE AND FILL HISTOGRAMS

column_names = ["avg"]

for col_name in column_names:
    # Create a unique histogram name
    hist_name = f"h_{col_name}"
    hist_title = f"Distribution of {col_name};Value;Counts"

    # Determine min/max for the histogram from the data
    col_min = df[col_name].min()
    col_max = df[col_name].max()

    # Create the 1D histogram (TH1F for float data)
    hist1, hist2, hist3 = None, None, None
    
    logbins = np.unique(np.concatenate([np.logspace(-3, -1, num = int(n * 0.7) + 1, endpoint = False), np.logspace(-1, np.log10(np.pi), num = int(n * 0.3), endpoint = True)]))
    logbins = np.array(np.sort(logbins), dtype=float)
    hist1 = ROOT.TH1F(hist_name, hist_title, n_bins, minn, maxx)
    hist2 = ROOT.TH1F(hist_name + 'Unif', hist_title, n_bins, minn, maxx)
    hist3 = ROOT.TH1F(hist_name + 'Corr', hist_title, n_bins, minn, maxx)
    canvas.SetLogy(1)
    # canvas.SetLogx(1)
    hist2.SetLineColor(ROOT.kRed) # Assign a unique color
    hist3.SetLineColor(ROOT.kGreen + 2)
    hist1.SetStats(False)
    
    legend = ROOT.TLegend(0.65, 0.8, 0.8, 0.9, "") 
    legend.AddEntry(hist1, "Original")
    legend.AddEntry(hist2, "Uniform")
    legend.AddEntry(hist3, "Correlated")
    
    # Fill the histogram efficiently using NumPy arrays
    # Ensure data is converted to a type ROOT can handle (e.g., NumPy array)
    data_array1 = df[col_name].dropna().to_numpy(dtype=float)
    data_array2 = df[col_name + 'Unif'].dropna().to_numpy(dtype=float)
    data_array3 = df[col_name + 'Corr'].dropna().to_numpy(dtype=float)
    
    
    # Loop through the data and fill the histogram
    for val1 in data_array1:
        hist1.Fill(val1)
    for val2 in data_array2:
        hist2.Fill(val2)
    for val3 in data_array3:
        hist3.Fill(val3)
    
    print("Chi2 test:", hist1.Chi2Test(hist3,"NORM"))
    
    hist1.Scale(1/hist1.GetEntries())
    hist2.Scale(1/hist2.GetEntries())
    hist3.Scale(1/hist3.GetEntries())
    
    stack = ROOT.THStack("hs", "")
    
    for h in [hist1, hist2, hist3]:
        stack.Add(h)
    stack.SetMaximum(1.0)
    stack.SetMinimum(0.000001)
    
    if "min" in col_name:
        stack.SetTitle("Distribution of min dPhi;DeltaPhi;Counts")
    if "avg" in col_name:
        stack.SetTitle("Distribution of avg dPhi;DeltaPhi;Counts")
    else:
        stack.SetTitle("Distribution of max dPhi;DeltaPhi;Counts")

    stack.Draw('NOSTACKhist')
    # 4. DRAW AND SAVE
    '''
    hist1.Draw('hist')
    hist2.Draw('histsame')
    hist3.Draw('histsame')'''
    legend.Draw('same')

    canvas.Update()
    
    # Print to the open multi-page PDF file
    canvas.SaveAs(output_canvas_name)
    
    # Prevent PyROOT from destroying the histogram object after the loop iteration
    # This is a common practice to avoid issues with Python's garbage collector.
    ROOT.gROOT.Append(hist1)
    ROOT.gROOT.Append(hist2)
    ROOT.gROOT.Append(hist3) 

# 5. FINAL CLEANUP

# Close the multi-page PDF file
#canvas.Print(f"{output_canvas_name}]") # The ']' closes the multi-page PDF

print("-" * 40)
print(f"SUCCESS: {len(column_names)} histograms saved to '{output_canvas_name}'")
print(f"Data Source: {input_filename}")
print("-" * 40)