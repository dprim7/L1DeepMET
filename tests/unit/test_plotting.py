import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from l1deepmet.plotting import (
    setup_cms_style,
    convertXY2PtPhi,
    resolqt,
    phidiff,
    Make1DHists,
    MakeEdgeHist,
    MET_rel_error_opaque,
    histo_2D,
    create_histogram_plots,
    create_correlation_plot
)


class TestPlottingFunctions:
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        n_events = 1000
        
        # Generate sample MET data
        true_xy = np.random.normal(0, 50, (n_events, 2))
        ml_xy = true_xy + np.random.normal(0, 10, (n_events, 2))
        puppi_xy = true_xy + np.random.normal(0, 15, (n_events, 2))
        
        return true_xy, ml_xy, puppi_xy
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    def test_setup_cms_style(self):
        """Test CMS style setup"""
        # Should not raise any errors
        setup_cms_style()
        assert plt.style.available  # Just check that styles are available
    
    def test_convertXY2PtPhi(self, sample_data):
        """Test XY to PtPhi conversion"""
        true_xy, _, _ = sample_data
        
        result = convertXY2PtPhi(true_xy)
        
        assert result.shape == true_xy.shape
        assert np.all(result[:, 0] >= 0)  # Pt should be positive
        assert np.all(result[:, 1] >= -np.pi) and np.all(result[:, 1] <= np.pi)  # Phi in [-π, π]
        
        # Test specific case
        test_xy = np.array([[3, 4], [0, 5], [-3, 0]])
        expected_pt = np.array([5, 5, 3])
        expected_phi = np.array([np.arctan2(4, 3), np.pi/2, np.pi])
        
        result = convertXY2PtPhi(test_xy)
        np.testing.assert_allclose(result[:, 0], expected_pt, rtol=1e-10)
        np.testing.assert_allclose(result[:, 1], expected_phi, rtol=1e-10)
    
    def test_resolqt(self):
        """Test resolution calculation"""
        # Test with normal distribution
        data = np.random.normal(0, 1, 1000)
        resolution = resolqt(data)
        
        assert resolution > 0
        assert abs(resolution - 1.0) < 0.2  # Should be close to 1 for normal distribution
        
        # Test with constant data
        constant_data = np.ones(100)
        resolution = resolqt(constant_data)
        assert resolution == 0.0
    
    def test_phidiff(self):
        """Test phi difference calculation"""
        # Test normal case
        assert abs(phidiff(0.5, 0.3) - 0.2) < 1e-10
        
        # Test wrapping around π
        assert abs(phidiff(0.1, 2*np.pi - 0.1) - 0.2) < 1e-10
        
        # Test wrapping around -π
        result = phidiff(-0.1, -2*np.pi + 0.1)
        # Verify the result is in the correct range
        assert -np.pi <= result <= np.pi
        # The smallest angular difference between -0.1 and 0.1 should be 0.2
        assert abs(abs(result) - 0.2) < 1e-10
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_make1d_hists(self, mock_close, mock_savefig, sample_data, temp_dir):
        """Test 1D histogram creation"""
        true_xy, ml_xy, puppi_xy = sample_data
        
        output_path = os.path.join(temp_dir, "test_hist.png")
        
        # Test basic functionality
        Make1DHists(
            true_xy[:, 0], ml_xy[:, 0], puppi_xy[:, 0],
            xmin=-100, xmax=100, nbins=20,
            xname="MET X [GeV]", yname="Events",
            outputname=output_path
        )
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_make_edge_hist(self, mock_close, mock_savefig, temp_dir):
        """Test edge histogram creation"""
        edge_data = np.random.normal(0, 1, 500)
        output_path = os.path.join(temp_dir, "test_edge_hist.png")
        
        MakeEdgeHist(
            edge_data, "Edge Feature", output_path,
            nbins=50, density=True, yname="Density"
        )
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_met_rel_error_opaque(self, mock_close, mock_show, mock_savefig, sample_data, temp_dir):
        """Test MET relative error plot"""
        true_xy, ml_xy, puppi_xy = sample_data
        
        # Convert to pt
        true_pt = convertXY2PtPhi(true_xy)[:, 0]
        ml_pt = convertXY2PtPhi(ml_xy)[:, 0]
        puppi_pt = convertXY2PtPhi(puppi_xy)[:, 0]
        
        output_path = os.path.join(temp_dir, "test_rel_error.pdf")
        
        MET_rel_error_opaque(ml_pt, puppi_pt, true_pt, output_path)
        
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
        mock_close.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_histo_2d(self, mock_close, mock_show, mock_savefig, sample_data, temp_dir):
        """Test 2D histogram creation"""
        true_xy, ml_xy, _ = sample_data
        
        true_pt = convertXY2PtPhi(true_xy)[:, 0]
        ml_pt = convertXY2PtPhi(ml_xy)[:, 0]
        
        output_path = os.path.join(temp_dir, "test_2d_hist.png")
        
        histo_2D(ml_pt, true_pt, 0, 200, output_path)
        
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
        mock_close.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_histogram_plots(self, mock_close, mock_savefig, temp_dir):
        """Test histogram plot creation"""
        features = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(1, 1000),
            'feature3': np.random.uniform(-5, 5, 1000)
        }
        
        create_histogram_plots(features, temp_dir, bins=50)
        
        # Should create individual plots + combined plot
        expected_calls = len(features) + 1
        assert mock_savefig.call_count == expected_calls
        assert mock_close.call_count == expected_calls
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_correlation_plot(self, mock_close, mock_savefig, temp_dir):
        """Test correlation plot creation"""
        features = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000) + 0.5 * np.random.normal(0, 1, 1000),  # Correlated
            'feature3': np.random.uniform(-5, 5, 1000)  # Uncorrelated
        }
        
        create_correlation_plot(features, temp_dir)
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_create_correlation_plot_single_feature(self, temp_dir):
        """Test correlation plot with single feature (should do nothing)"""
        features = {'feature1': np.random.normal(0, 1, 100)}
        
        # Should not raise error and should not create plot
        create_correlation_plot(features, temp_dir)
        
        # Check that no files were created
        files = list(Path(temp_dir).glob('*'))
        assert len(files) == 0
    
    def test_analyze_outliers(self, temp_dir):
        """Test outlier analysis"""
        features = {
            'normal': np.random.normal(0, 1, 1000),
            'with_outliers': np.concatenate([
                np.random.normal(0, 1, 950),
                np.random.normal(10, 1, 50)  # Outliers
            ])
        }
        
        from l1deepmet.plotting import analyze_outliers
        analyze_outliers(features, temp_dir)
        
        # Check that analysis file was created
        analysis_file = os.path.join(temp_dir, 'outlier_analysis.txt')
        assert os.path.exists(analysis_file)
        
        # Check file content
        with open(analysis_file, 'r') as f:
            content = f.read()
            assert 'Feature Outlier Analysis' in content
            assert 'normal:' in content
            assert 'with_outliers:' in content
