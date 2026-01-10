import AsyncStorage from '@react-native-async-storage/async-storage';
import {useIsFocused} from '@react-navigation/native';
import {Camera, CameraView} from 'expo-camera';
import * as Notifications from 'expo-notifications';
import {useRouter} from 'expo-router';
import React, {useContext, useEffect, useRef, useState} from 'react';
import {Alert, Button, Modal, StyleSheet, Text, TextInput, View} from 'react-native';
import {UserContext} from '../userContext';

export default function App() {
    const [hasPermission, setHasPermission] = useState<boolean | null>(null);
    const [scanned, setScanned] = useState(false);
    const [barcodeData, setBarcodeData] = useState('');
    const [productName, setProductName] = useState('');
    const scanLockRef = useRef(false);
    const {user} = useContext(UserContext)!;
    const router = useRouter();
    const isFocused = useIsFocused();
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [editedName, setEditedName] = useState('');
    const [editedWeight, setEditedWeight] = useState('');
    const [editedUnit, setEditedUnit] = useState('');
    const [saveName, setSaveName] = useState('');
    const [saveUnit, setSaveUnit] = useState('');
    const [saveWeight, setSaveWeight] = useState('');

    const API_BASE_URL = 'http://localhost:3001';

    useEffect(() => {
        const subscriptionReceived = Notifications.addNotificationReceivedListener(notification => {
            console.log('Notifikacija:', notification);
        });

        const subscriptionResponse = Notifications.addNotificationResponseReceivedListener(response => {
            console.log('Obvestilo:', response);
            const data = response.notification.request.content.data;

            if (data?.type === 'login_confirmation') {
                router.push('/faceAuth');
            }
        });

        return () => {
            subscriptionReceived.remove();
            subscriptionResponse.remove();
        };
    }, [router]);

    useEffect(() => {
        const registerForPushNotifications = async () => {
            try {
                const {status: existingStatus} = await Notifications.getPermissionsAsync();
                let finalStatus = existingStatus;
                if (existingStatus !== 'granted') {
                    const {status} = await Notifications.requestPermissionsAsync();
                    finalStatus = status;
                }
                if (finalStatus !== 'granted') {
                    console.warn('Push notification permissions not granted!');
                    return;
                }
                const pushtoken = await Notifications.getExpoPushTokenAsync();
                const token = await AsyncStorage.getItem('token');
                const userId = await AsyncStorage.getItem('userId');
                if (userId && pushtoken?.data) {
                    await fetch(`${API_BASE_URL}/users/savePushToken`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`,
                        },
                        body: JSON.stringify({
                            user: userId,
                            token: pushtoken.data,
                        }),
                    });
                }
            } catch (error) {
                console.error("Error registering push notifications:", error);
            }
        };
        registerForPushNotifications();
    }, []);

    useEffect(() => {
        const getCameraPermissions = async () => {
            const {status} = await Camera.requestCameraPermissionsAsync();
            setHasPermission(status === 'granted');
        };
        getCameraPermissions();
    }, []);

    useEffect(() => {
        if (!user) {
            router.replace('../login');
        }
    }, [user]);

    const fetchProductData = async (barcode: string) => {
        try {
            const localRes = await fetch(`${API_BASE_URL}/barcodes/${barcode}`);
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

    const handleBarcodeScanned = ({type, data}: {type: string, data: string}) => {
        if (scanLockRef.current) return;
        scanLockRef.current = true;
        setScanned(true);
        setBarcodeData(data);
        setProductName('Loading...');
        fetchProductData(data);
    };

    const resetScanner = () => {
        scanLockRef.current = false;
        setScanned(false);
        setBarcodeData('');
        setProductName('');
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
            const response = await fetch(`${API_BASE_URL}/barcodes`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
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
        
        if (!saveName) {
            Alert.alert('Error', 'No product data to add.');
            return;
        }

        const token = await AsyncStorage.getItem('token');
        try {
            const response = await fetch(`${API_BASE_URL}/myfridge/barcodeScan`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json', Authorization: `Bearer ${token}`},
                body: JSON.stringify(ingredient),
            });

            if (!response.ok) {
                throw new Error(`Failed to add to fridge: ${response.status}`);
            }

            await response.json();
            Alert.alert('Success', 'Ingredient added to your fridge!');
            setScanned(false);
            setProductName('');
            scanLockRef.current = false;
        } catch (error) {
            console.error('Error adding to fridge:', error);
            Alert.alert('Error', 'Failed to add to fridge.');
        }
    };

    if (hasPermission === null) {
        return <View style={styles.container}><Text>Requesting for camera permission</Text></View>;
    }
    if (hasPermission === false) {
        return <View style={styles.container}><Text>No access to camera</Text></View>;
    }

    return (
        <View style={styles.container}>
            {isFocused && (
                <CameraView
                    style={StyleSheet.absoluteFillObject}
                    onBarcodeScanned={scanned ? undefined : handleBarcodeScanned}
                    barcodeScannerSettings={{
                        barcodeTypes: ['qr', 'ean13', 'ean8', 'upc_a', 'upc_e'],
                    }}
                />
            )}
            {/* ... Rest of your UI components ... */}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
    },
});
