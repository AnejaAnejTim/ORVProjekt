import AsyncStorage from '@react-native-async-storage/async-storage';
import { useIsFocused } from '@react-navigation/native';
import { Camera, CameraView } from 'expo-camera';
import { useRouter } from 'expo-router';
import React, { useContext, useEffect, useRef, useState } from 'react';
import { Alert, Button, Modal, StyleSheet, Text, TextInput, View } from 'react-native';
import { UserContext } from '../userContext';

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [scanned, setScanned] = useState(false);
  const [barcodeData, setBarcodeData] = useState('');
  const [productName, setProductName] = useState('');
  const scanLockRef = useRef(false);
  const { user } = useContext(UserContext)!;
  const router = useRouter();
  const isFocused = useIsFocused();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editedName, setEditedName] = useState('');
  const [editedWeight, setEditedWeight] = useState('');
  const [editedUnit, setEditedUnit] = useState('');
  const [saveName, setSaveName] = useState('');
  const [saveUnit, setSaveUnit] = useState('');
  const [saveWeight, setSaveWeight] = useState('');

  useEffect(() => {
    const getCameraPermissions = async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    };

    getCameraPermissions();
  }, []);

  useEffect(() => {
    if (!user) {
      router.replace('../login');
    }
  }, [user]);

  const fetchProductData = async (barcode) => {
    try {
      const localRes = await fetch(`http://100.102.9.9:3001/barcodes/${barcode}`);
      if (localRes.ok) {
        const localData = await localRes.json();
        const name = localData.product_name || 'Unnamed product';
        const unit = localData.unit || '';
        const weight = localData.weight || '';
        setSaveName(name);
        setSaveUnit(unit);
        setSaveWeight(weight);
        setProductName(`${name} ${weight}${unit}`);
        return;
      }

      const response = await fetch(`https://world.openfoodfacts.org/api/v0/product/${barcode}.json?lc=sl`);
      const json = await response.json();
      if (json.status === 1) {
        const name = json.product.product_name_sl ||
                     json.product.product_name_en ||
                     json.product.product_name ||
                     'Unnamed product';
        const packaging = json.product.packagings?.[0];
        const quantity = packaging?.quantity_per_unit_value || '';
        const unit = packaging?.quantity_per_unit_unit || '';
        setSaveName(name);
        setSaveUnit(unit);
        setSaveWeight(quantity);
        setProductName(`${name} ${quantity}${unit}`);
      } else {
        setProductName('Product not found');
      }
    } catch (err) {
      console.error('API error:', err);
      setProductName('Error fetching product');
    }
  };

  const handleBarcodeScanned = ({ type, data }) => {
    if (scanLockRef.current) return;
    scanLockRef.current = true;
    setScanned(true);
    setBarcodeData(data);
    setProductName('');
    fetchProductData(data);
  };

  const handleEditPress = () => {
    const [namePart, quantityPart] = productName.split(/ (?=\d)/);
    const match = quantityPart?.match(/^(\d+)([a-zA-Z]+)?$/);
    setEditedName(namePart || '');
    setEditedWeight(match?.[1] || '');
    setEditedUnit(match?.[2] || '');
    setIsModalVisible(true);
  };

  const handleSaveEdit = async () => {
    const newDisplayName = `${editedName} ${editedWeight}${editedUnit}`;
    setSaveName(editedName);
    setSaveUnit(editedUnit);
    setSaveWeight(editedWeight);
    setProductName(newDisplayName);
    setIsModalVisible(false);

    try {
      const response = await fetch('http://100.102.9.9:3001/barcodes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: barcodeData,
          product_name: editedName,
          weight: editedWeight,
          unit: editedUnit,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save: ${response.status}`);
      }

      await response.json();
    } catch (error) {
      console.error('Error saving product:', error);
      Alert.alert('Error', 'Failed to save product. Please try again.');
    }
  };

  const handleAddPress = async () => {
    const ingredient = {
      name: saveName,
      unit: saveUnit,
      quantity: saveWeight,
    };
      const token = await AsyncStorage.getItem('token');
      console.log('Token before myfridge fetch:', token);
    try {
      const response = await fetch('http://100.102.9.9:3001/myfridge/barcodeScan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json',Authorization: `Bearer ${token}`, },
        body: JSON.stringify(ingredient),
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`Failed to add to fridge: ${response.status}`);
      }

      await response.json();
      Alert.alert('Success', 'Ingredient added to your fridge!');
      setScanned(false);
      scanLockRef.current = false;
      setBarcodeData('');
      setProductName('');
    } catch (error) {
      console.error('Error adding ingredient:', error);
      Alert.alert('Error', 'Failed to add ingredient. Please try again.');
    }
  };

  if (hasPermission === null) return <Text>Requesting for camera permission</Text>;
  if (hasPermission === false) return <Text>No access to camera</Text>;

  return (
    <View style={styles.container}>
      {isFocused && (
        <CameraView
          style={StyleSheet.absoluteFillObject}
          facing="back"
          onBarcodeScanned={scanned ? undefined : handleBarcodeScanned}
          barcodeScannerSettings={{
            barcodeTypes: [
              'qr', 'pdf417', 'aztec', 'ean13', 'ean8', 'upc_a', 'upc_e',
              'code128', 'code39', 'code93', 'codabar', 'itf14', 'datamatrix',
            ],
          }}
        />
      )}

      {/* Modal for editing product info */}
      <Modal
        visible={isModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setIsModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Spremeni produkt</Text>
            <TextInput placeholder="Ime" value={editedName} onChangeText={setEditedName} style={styles.input} />
            <TextInput placeholder="Količina" value={editedWeight} onChangeText={setEditedWeight} keyboardType="numeric" style={styles.input} />
            <TextInput placeholder="Enota (npr. g, ml)" value={editedUnit} onChangeText={setEditedUnit} style={styles.input} />
            <View style={styles.modalButtons}>
              <Button title="Cancel" color="#aaa" onPress={() => setIsModalVisible(false)} />
              <Button title="Save" onPress={handleSaveEdit} />
            </View>
          </View>
        </View>
      </Modal>

      <View style={styles.overlay}>
        <Text style={styles.title}>Skeniraj svoj produkt</Text>
        {productName && (
          <View style={styles.resultBox}>
            <Text style={[styles.resultText, { marginTop: 10, fontWeight: 'bold', color: '#0a0' }]}>
              {productName}
            </Text>
            <View style={styles.buttonsRow}>
              <View style={styles.buttonWrapper}>
                <Button title="Edit" onPress={handleEditPress} color="#007BFF" />
              </View>
              <View style={styles.buttonWrapper}>
                <Button title="Add!" onPress={handleAddPress} color="#28a745" disabled={saveName === 'Product not found'} />
              </View>
            </View>
          </View>
        )}
        {scanned && (
          <View style={styles.scanAgainButton}>
            <Button title="Scan Again" onPress={() => {
              setScanned(false);
              scanLockRef.current = false;
              setBarcodeData('');
              setProductName('');
            }} color="#fff" />
          </View>
        )}
      </View>
    </View>
  );
}


const styles = StyleSheet.create({
  modalOverlay: {
  flex: 1,
  backgroundColor: 'rgba(0, 0, 0, 0.6)',
  justifyContent: 'center',
  alignItems: 'center',
  padding: 20,
},
modalContent: {
  width: '90%',
  backgroundColor: '#fff',
  borderRadius: 10,
  padding: 20,
  shadowColor: '#000',
  shadowOpacity: 0.25,
  shadowOffset: { width: 0, height: 2 },
  shadowRadius: 4,
  elevation: 5,
},
modalTitle: {
  fontSize: 18,
  fontWeight: 'bold',
  marginBottom: 10,
  textAlign: 'center',
},
input: {
  borderWidth: 1,
  borderColor: '#ccc',
  borderRadius: 5,
  padding: 10,
  marginBottom: 10,
},
modalButtons: {
  flexDirection: 'row',
  justifyContent: 'space-between',
  marginTop: 10,
},
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
