import React, { useState, useEffect, useRef } from 'react';
import { Text, View, StyleSheet, Button, Alert } from 'react-native';
import { CameraView, Camera } from 'expo-camera';

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [scanned, setScanned] = useState(false);
  const [barcodeData, setBarcodeData] = useState('');
  const [productName, setProductName] = useState('');
  const scanLockRef = useRef(false); // To prevent rapid scans

  useEffect(() => {
    const getCameraPermissions = async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    };

    getCameraPermissions();
  }, []);

  const fetchProductData = async (barcode) => {
    try {
      console.log(`Fetching product data for barcode: ${barcode}`);
      const response = await fetch(`https://world.openfoodfacts.org/api/v0/product/${barcode}.json`);
      const json = await response.json();
      console.log('API response:', json);

      if (json.status === 1) {
        const name = json.product.product_name || 'Unnamed product';
        console.log('Product found:', name);
        setProductName(name);
      } else {
        console.log('Product not found in Open Food Facts');
        setProductName('Product not found');
      }
    } catch (err) {
      console.error('API error:', err);
      setProductName('Error fetching product');
    }
  };

  const handleBarcodeScanned = ({ type, data }) => {
    if (scanLockRef.current) return; // Ignore if locked

    scanLockRef.current = true;
    setScanned(true);
    setBarcodeData(data);
    setProductName('');
    fetchProductData(data);
  };

  const handleEditPress = () => {
    Alert.alert('Edit button pressed', `Edit product: ${productName}`);
  };

  const handleAddPress = () => {
    Alert.alert('Add button pressed', `Add product: ${productName}`);
  };

  if (hasPermission === null) {
    return <Text>Requesting for camera permission</Text>;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={StyleSheet.absoluteFillObject}
        facing="back"
        onBarcodeScanned={scanned ? undefined : handleBarcodeScanned}
        barcodeScannerSettings={{
          barcodeTypes: [
            'qr',
            'pdf417',
            'aztec',
            'ean13',
            'ean8',
            'upc_a',
            'upc_e',
            'code128',
            'code39',
            'code93',
            'codabar',
            'itf14',
            'datamatrix',
          ],
        }}
      />

      <View style={styles.overlay}>
        <Text style={styles.title}>Barcode Scanner</Text>

        
          {productName ? (
            <View style={styles.resultBox}>
            <>
              <Text style={[styles.resultText, { marginTop: 10, fontWeight: 'bold', color: '#0a0' }]}>
                {productName}
              </Text>

              <View style={styles.buttonsRow}>
                <View style={styles.buttonWrapper}>
                  <Button title="Edit" onPress={handleEditPress} color="#007BFF" />
                </View>
                <View style={styles.buttonWrapper}>
                  <Button title="Add!" onPress={handleAddPress} color="#28a745" />
                </View>
              </View>
            </>
            </View>
          ) : null}
        

        {scanned && (
          <View style={styles.scanAgainButton}>
            <Button
              title="Scan Again"
              onPress={() => {
                setScanned(false);
                scanLockRef.current = false;
                setBarcodeData('');
                setProductName('');
              }}
              color="#fff"
            />
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 80,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 10,
  },
  resultBox: {
    backgroundColor: 'rgba(255,255,255,0.9)',
    padding: 20,
    borderRadius: 10,
    marginVertical: 20,
    minWidth: 300,
    alignItems: 'center',
  },
  resultText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#333',
  },
  buttonsRow: {
    flexDirection: 'row',
    marginTop: 15,
    justifyContent: 'space-around',
    width: '80%',
  },
  buttonWrapper: {
    flex: 1,
    marginHorizontal: 5,
  },
  scanAgainButton: {
    width: '100%',
    marginBottom: 10,
    backgroundColor: '#007BFF',
    borderRadius: 5,
  },
});
