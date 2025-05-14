import { Button, StyleSheet, Text, TextInput, View, Modal, Alert, Platform } from 'react-native';
import { CameraView, CameraType, useCameraPermissions, BarCodeScanningResult } from 'expo-camera';
import { useEffect, useState } from 'react';

export default function Index() {
  const [camType, setCamType] = useState<CameraType>('back');
  const [perm, requestPerm] = useCameraPermissions();
  const [scanned, setScanned] = useState(false);
  const [code, setCode] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [product, setProduct] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);
  const [debugMsg, setDebugMsg] = useState('');

  useEffect(() => {
    console.log("Camera permission status:", perm);
    if (!perm?.granted) {
      console.log("Requesting camera permission");
      requestPerm();
    } else {
      console.log("Camera permission already granted");
    }

    if (Platform.OS === 'ios') {
      console.log("Running on iOS - checking camera configuration");
    }
  }, [perm]);

  const handleScan = async ({ data, type }: BarCodeScanningResult) => {
    console.log("Barcode scan detected!", { data, type });
    setDebugMsg(`Scanned: ${type} - ${data}`);
    setScanned(true);
    setCode(data);

    try {
      console.log(`Fetching product data for code: ${data}`);
      const res = await fetch(`https://world.openfoodfacts.org/api/v0/product/${data}.json`);
      const json = await res.json();
      console.log("API response status:", json.status);

      if (json.status === 1) {
        console.log("Product found:", json.product.product_name);
        setName(json.product.product_name || '');
        setProduct(json.product);
        setShowModal(true);
      } else {
        console.log("Product not found");
        Alert.alert("Not found", "Manually enter product name", [
          {
            text: "OK",
            onPress: () => {
              setName('');
              setProduct(null);
              setShowModal(true);
            },
          },
        ]);
      }
    } catch (err) {
      console.error("API fetch error:", err);
      setDebugMsg(`Error: ${err.message}`);
      Alert.alert("Error", "Error gathering the data.");
    }
  };

  if (!perm) return <Text>Gathering permissions...</Text>;
  if (!perm.granted) {
    return (
      <View style={styles.center}>
        <Text>Camera permission is required for barcode scanning</Text>
        <Button title="Grant Camera Permission" onPress={requestPerm} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing={camType}
        barcodeScannerSettings={{
          barCodeTypes: ['ean13', 'ean8', 'upc_a', 'upc_e', 'qr', 'code39', 'code128']
        }}
        onBarcodeScanned={scanned ? undefined : handleScan}
      />

      {/* Debug overlay */}
      <View style={styles.debugOverlay}>
        <Text style={styles.debugText}>
          {debugMsg || 'Waiting for barcode...'}
        </Text>
        {code && <Text style={styles.debugText}>Last code: {code}</Text>}
      </View>

      <View style={styles.buttons}>
        <Button
          title="Flip camera"
          onPress={() => setCamType((t) => (t === 'back' ? 'front' : 'back'))}
        />
        {scanned && <Button title="Scan again" onPress={() => setScanned(false)} />}
      </View>

      <Modal visible={showModal} animationType="slide" transparent>
        <View style={styles.overlay}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>Is this the correct product?</Text>
            <TextInput
              style={styles.input}
              value={name}
              onChangeText={setName}
              placeholder="Product name"
            />
            <View style={styles.modalButtons}>
              <Button title="Confirm" onPress={() => setShowModal(false)} />
              <View style={{ width: 10 }} />
              <Button
                title="Cancel"
                color="red"
                onPress={() => {
                  setShowModal(false);
                  setScanned(false);
                  setCode(null);
                  setName('');
                  setProduct(null);
                }}
              />
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  buttons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 10,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modal: {
    backgroundColor: '#fff',
    padding: 20,
    width: '80%',
    borderRadius: 10,
    elevation: 5,
  },
  modalTitle: {
    fontSize: 18,
    marginBottom: 10,
    fontWeight: 'bold',
  },
  input: {
    borderWidth: 1,
    borderColor: '#aaa',
    padding: 10,
    borderRadius: 5,
    marginBottom: 15,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  debugOverlay: {
    position: 'absolute',
    top: 40,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 10,
  },
  debugText: {
    color: 'white',
    fontSize: 12,
  }
});