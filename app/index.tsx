import { Button, StyleSheet, Text, TextInput, View, Alert, Platform, FlatList, ScrollView, TouchableOpacity } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useEffect, useState, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function Index() {
  const [camType, setCamType] = useState('back');
  const [perm, requestPerm] = useCameraPermissions();
  const [scanned, setScanned] = useState(false);
  const [code, setCode] = useState(null);
  const [name, setName] = useState('');
  const [quantity, setQuantity] = useState('');
  const [producer, setProducer] = useState('');
  const [product, setProduct] = useState(null);
  const [showSavedProducts, setShowSavedProducts] = useState(false);
  const [showProductDetails, setShowProductDetails] = useState(false);
  const [showProductInput, setShowProductInput] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [debugMsg, setDebugMsg] = useState('');
  const [scanAttempts, setScanAttempts] = useState(0);
  const [savedProducts, setSavedProducts] = useState([]);

  useEffect(() => {
    loadSavedProducts();
  }, []);

  const loadSavedProducts = async () => {
    try {
      const productsJson = await AsyncStorage.getItem('savedProducts');
      if (productsJson) {
        const products = JSON.parse(productsJson);
        setSavedProducts(products);
        console.log(`Loaded ${products.length} saved products`);
      }
    } catch (error) {
      console.error('Error loading saved products:', error);
      setDebugMsg(`Storage error: ${error.message}`);
    }
  };

  const saveProductsToStorage = async (updatedProducts) => {
    try {
      await AsyncStorage.setItem('savedProducts', JSON.stringify(updatedProducts));
      console.log(`Saved ${updatedProducts.length} products to storage`);
    } catch (error) {
      console.error('Error saving products:', error);
      setDebugMsg(`Save error: ${error.message}`);
    }
  };

  const addProduct = async (barcode, productName, productData = null, productQuantity = '', productProducer = '') => {
    const newProduct = {
      id: barcode,
      name: productName,
      quantity: productQuantity,
      producer: productProducer,
      timestamp: new Date().toISOString(),
      apiData: productData || {}
    };

    const updatedProducts = [...savedProducts, newProduct];
    setSavedProducts(updatedProducts);
    await saveProductsToStorage(updatedProducts);

    console.log(`Added product: ${productName} (${barcode})`);
    return newProduct;
  };

  useEffect(() => {
    console.log("Camera permission status:", perm);
    if (!perm?.granted) {
      console.log("Requesting camera permission");
      requestPerm();
    } else {
      console.log("Camera permission already granted");
    }
  }, [perm]);

  useEffect(() => {
    if (scanAttempts > 0) {
      console.log(`Total scan attempts: ${scanAttempts}`);
    }
  }, [scanAttempts]);

  const findProductByBarcode = useCallback((barcode) => {
    return savedProducts.find(p => p.id === barcode);
  }, [savedProducts]);

  const findSimilarProduct = useCallback((barcode) => {
    if (!barcode || barcode.length < 8) return null;

    const prefix = barcode.substring(0, 8);
    return savedProducts.find(p => p.id.startsWith(prefix) && p.id !== barcode);
  }, [savedProducts]);

  const processBarcode = useCallback(async (barcode) => {
    try {
      console.log(`Processing barcode: ${barcode}`);
      setDebugMsg(`Processing: ${barcode}`);

      const existingProduct = findProductByBarcode(barcode);

      if (existingProduct) {
        console.log("Product found in local storage:", existingProduct.name);
        setName(existingProduct.name || '');
        setQuantity(existingProduct.quantity || '');
        setProducer(existingProduct.producer || '');
        setProduct(existingProduct);
        setShowProductInput(true);
        return;
      }

      const similarProduct = findSimilarProduct(barcode);
      if (similarProduct) {
        console.log("Similar product found:", similarProduct.name);
        Alert.alert(
          "Similar Product Found",
          `We found a similar product: "${similarProduct.name}". Is this the same product?`,
          [
            {
              text: "Yes, use this",
              onPress: () => {
                setName(similarProduct.name);
                setQuantity(similarProduct.quantity || '');
                setProducer(similarProduct.producer || '');
                setProduct({ ...similarProduct, id: barcode });
                setShowProductInput(true);
              }
            },
            {
              text: "No, enter manually",
              onPress: () => {
                setName('');
                setQuantity('');
                setProducer('');
                setProduct(null);
                setShowProductInput(true);
              }
            }
          ]
        );
        return;
      }

      const res = await fetch(`https://world.openfoodfacts.org/api/v0/product/${barcode}.json`);
      const json = await res.json();

      if (json.status === 1 && json.product) {
        const prod = json.product;

        const name = prod.product_name || 'Unnamed Product';
        const quantity = prod.quantity || (prod.packaging_quantity || '');
        const producer = prod.brands || prod.manufacturer || '';

        console.log("Product found in API:", name);
        setName(name);
        setQuantity(quantity);
        setProducer(producer);
        setProduct(prod);
        setShowProductInput(true);

        await addProduct(
          barcode,
          name,
          prod,
          quantity,
          producer
        );
      } else {
        console.log("Product not found in API");
        Alert.alert("Product Not Found", "Please enter product details manually", [
          {
            text: "OK",
            onPress: () => {
              setName('');
              setQuantity('');
              setProducer('');
              setProduct(null);
              setShowProductInput(true);
            },
          },
        ]);
      }
    } catch (err) {
      console.error('API fetch error:', err);
      setDebugMsg(`API Error: ${err.message}`);

      Alert.alert("Connection Error", "Please enter product details manually", [
        {
          text: "OK",
          onPress: () => {
            setName('');
            setQuantity('');
            setProducer('');
            setProduct(null);
            setShowProductInput(true);
          },
        },
      ]);
    }
  }, [findProductByBarcode, findSimilarProduct, savedProducts]);

  const handleScan = useCallback((result) => {
    console.log('SCAN EVENT RECEIVED:', JSON.stringify(result));
    setScanAttempts(prev => prev + 1);

    const { data, type } = result || {};

    setDebugMsg(`Scan attempt: ${type || 'unknown'}`);

    if (!data) {
      console.warn('Empty or invalid barcode data received');
      return;
    }

    console.log(`Valid barcode scanned: ${type} - ${data}`);
    setDebugMsg(`Scanned: ${type} - ${data}`);

    setScanned(true);
    setCode(data);

    processBarcode(data);
  }, [processBarcode]);

  const handleSaveProduct = useCallback(async () => {
    if (!code || !name.trim()) {
      Alert.alert("Error", "Product code and name are required");
      return;
    }

    // Check if we're updating an existing product
    const existingProduct = findProductByBarcode(code);

    if (existingProduct) {
      // Update existing product, preserving apiData
      const updatedProducts = savedProducts.map(p =>
        p.id === code ? {
          ...p,
          name: name,
          quantity: quantity,
          producer: producer,
          timestamp: new Date().toISOString()
          // Keep existing apiData if present
        } : p
      );
      setSavedProducts(updatedProducts);
      await saveProductsToStorage(updatedProducts);
      console.log(`Updated product: ${name} (${code})`);
    } else {
      // Add new product
      await addProduct(code, name, product, quantity, producer);
    }

    setShowProductInput(false);
    setScanned(false);
    setCode(null);
    setName('');
    setQuantity('');
    setProducer('');
    setProduct(null);
  }, [code, name, quantity, producer, product, savedProducts, findProductByBarcode]);

  const handleCancelSave = useCallback(() => {
    setShowProductInput(false);
    setScanned(false);
    setCode(null);
    setName('');
    setQuantity('');
    setProducer('');
    setProduct(null);
  }, []);

  const showProductDetailsScreen = (product) => {
    setSelectedProduct(product);
    setShowProductDetails(true);
  };

  const renderProductInputForm = () => {
    return (
      <View style={styles.formContainer}>
        <View style={styles.formHeader}>
          <Text style={styles.formTitle}>Product Info</Text>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={handleCancelSave}
          >
            <Text style={styles.closeButtonText}>✕</Text>
          </TouchableOpacity>
        </View>

        <TextInput
          style={styles.input}
          placeholder="Product Name"
          value={name}
          onChangeText={setName}
        />
        <TextInput
          style={styles.input}
          placeholder="Quantity"
          value={quantity}
          onChangeText={setQuantity}
        />
        <TextInput
          style={styles.input}
          placeholder="Producer"
          value={producer}
          onChangeText={setProducer}
        />
        <View style={styles.formButtons}>
          <TouchableOpacity
            style={[styles.button, styles.saveButton]}
            onPress={handleSaveProduct}
          >
            <Text style={styles.buttonText}>Save</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.cancelButton]}
            onPress={handleCancelSave}
          >
            <Text style={styles.buttonText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  const renderSavedProductItem = ({ item }) => (
    <TouchableOpacity
      style={styles.productItem}
      onPress={() => showProductDetailsScreen(item)}
    >
      <Text style={styles.productName}>{item.name}</Text>
      <Text style={styles.productCode}>Code: {item.id}</Text>
      {item.quantity ? <Text>Quantity: {item.quantity}</Text> : null}
      {item.producer ? <Text>Producer: {item.producer}</Text> : null}
      <Text style={styles.productDate}>
        Added: {new Date(item.timestamp).toLocaleDateString()}
      </Text>
      <Text style={styles.tapForMore}>Tap for more details</Text>
    </TouchableOpacity>
  );

  const renderProductDetails = () => {
    if (!selectedProduct) return null;

    const apiData = selectedProduct.apiData || {};

    const details = [
      { label: "Product Name", value: selectedProduct.name },
      { label: "Barcode", value: selectedProduct.id },
      { label: "Quantity", value: selectedProduct.quantity },
      { label: "Producer/Brand", value: selectedProduct.producer },
      { label: "Added On", value: new Date(selectedProduct.timestamp).toLocaleString() },
      { label: "Ingredients", value: apiData.ingredients_text },
      { label: "Allergens", value: apiData.allergens_tags?.join(", ").replace(/^en:/g, "") },
      { label: "Nutrition Grade", value: apiData.nutrition_grades?.toUpperCase() },
      { label: "Categories", value: apiData.categories },
      { label: "Origin", value: apiData.origin },
      { label: "Manufacturing Places", value: apiData.manufacturing_places },
      { label: "Stores", value: apiData.stores },
      { label: "Countries", value: apiData.countries },
      { label: "Labels", value: apiData.labels },
      { label: "Energy (100g)", value: apiData.nutriments?.energy_100g ? `${apiData.nutriments.energy_100g} ${apiData.nutriments.energy_unit}` : null },
      { label: "Fat (100g)", value: apiData.nutriments?.fat_100g ? `${apiData.nutriments.fat_100g}g` : null },
      { label: "Sugars (100g)", value: apiData.nutriments?.sugars_100g ? `${apiData.nutriments.sugars_100g}g` : null },
      { label: "Protein (100g)", value: apiData.nutriments?.proteins_100g ? `${apiData.nutriments.proteins_100g}g` : null },
      { label: "Salt (100g)", value: apiData.nutriments?.salt_100g ? `${apiData.nutriments.salt_100g}g` : null },
    ];

    const nutrientLevels = apiData.nutrient_levels || {};
    const nutrientDetails = Object.entries(nutrientLevels).map(([key, value]) => {
      const nutrientName = key.replace(/_/g, ' ');
      return { label: `${nutrientName.charAt(0).toUpperCase() + nutrientName.slice(1)}`, value };
    });

    return (
      <View style={styles.detailsContainer}>
        <ScrollView style={styles.detailsScroll}>
          <View style={styles.detailsHeader}>
            <Text style={styles.detailsTitle}>{selectedProduct.name}</Text>
            <Text style={styles.detailsBarcode}>{selectedProduct.id}</Text>
          </View>

          <View style={styles.detailsSection}>
            <Text style={styles.sectionTitle}>Basic Information</Text>
            {details.filter(item => item.value).map((item, index) => (
              <View key={index} style={styles.detailRow}>
                <Text style={styles.detailLabel}>{item.label}:</Text>
                <Text style={styles.detailValue}>{item.value}</Text>
              </View>
            ))}
          </View>

          {nutrientDetails.length > 0 && (
            <View style={styles.detailsSection}>
              <Text style={styles.sectionTitle}>Nutrient Levels</Text>
              {nutrientDetails.map((item, index) => (
                <View key={index} style={styles.detailRow}>
                  <Text style={styles.detailLabel}>{item.label}:</Text>
                  <Text
                    style={[
                      styles.detailValue,
                      item.value === 'high' ? styles.highLevel :
                      item.value === 'moderate' ? styles.moderateLevel :
                      styles.lowLevel
                    ]}
                  >
                    {item.value.charAt(0).toUpperCase() + item.value.slice(1)}
                  </Text>
                </View>
              ))}
            </View>
          )}

          {/* Additional sections for more detailed data */}
          {apiData.ingredients && (
            <View style={styles.detailsSection}>
              <Text style={styles.sectionTitle}>Ingredients Detail</Text>
              {apiData.ingredients.map((ingredient, idx) => (
                <Text key={idx} style={styles.ingredientText}>
                  • {ingredient.text}
                  {ingredient.percent ? ` (${ingredient.percent}%)` : ''}
                  {ingredient.vegan === 'yes' ? ' 🌱' : ''}
                  {ingredient.vegetarian === 'no' ? ' 🥩' : ''}
                </Text>
              ))}
            </View>
          )}

          <View style={styles.detailsSection}>
            <Text style={styles.sectionTitle}>Raw Data</Text>
            <TouchableOpacity
              style={styles.rawDataButton}
              onPress={() => Alert.alert("Raw API Data", JSON.stringify(apiData, null, 2))}
            >
              <Text style={styles.rawDataButtonText}>View Raw API Data</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>

        <Button
          title="Back to Products List"
          onPress={() => setShowProductDetails(false)}
        />
      </View>
    );
  };

  const clearAllProducts = async () => {
    Alert.alert(
      "Clear All Products",
      "Are you sure you want to delete all saved products?",
      [
        {
          text: "Cancel",
          style: "cancel"
        },
        {
          text: "Delete All",
          style: "destructive",
          onPress: async () => {
            try {
              await AsyncStorage.removeItem('savedProducts');
              setSavedProducts([]);
              console.log("All products cleared");
            } catch (error) {
              console.error("Error clearing products:", error);
              Alert.alert("Error", "Failed to clear products");
            }
          }
        }
      ]
    );
  };

  const deleteProduct = async (productId) => {
    Alert.alert(
      "Delete Product",
      "Are you sure you want to delete this product?",
      [
        {
          text: "Cancel",
          style: "cancel"
        },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              const updatedProducts = savedProducts.filter(p => p.id !== productId);
              setSavedProducts(updatedProducts);
              await saveProductsToStorage(updatedProducts);
              setShowProductDetails(false);
              console.log(`Product ${productId} deleted`);
            } catch (error) {
              console.error("Error deleting product:", error);
              Alert.alert("Error", "Failed to delete product");
            }
          }
        }
      ]
    );
  };

  if (!perm) return <Text>Gathering permissions...</Text>;
  if (!perm?.granted) {
    return (
      <View style={styles.center}>
        <Text>Camera permission is required for barcode scanning</Text>
        <Button title="Grant Camera Permission" onPress={requestPerm} />
      </View>
    );
  }

  // Product details screen
  if (showProductDetails) {
    return (
      <View style={styles.container}>
        {renderProductDetails()}
        <Button
          title="Delete This Product"
          onPress={() => deleteProduct(selectedProduct.id)}
          color="red"
        />
      </View>
    );
  }

  if (showProductInput) {
    return (
      <View style={styles.container}>
        {renderProductInputForm()}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {!showSavedProducts ? (
        <>
<CameraView
        style={styles.camera}
        barcodeScannerSettings={{
          barCodeTypes: ['ean13', 'ean8', 'upc_a', 'upc_e', 'code39', 'code128'],
          // iOS-specific optimizations
          ...(Platform.OS === 'ios' ? {
            isGuidanceEnabled: true,
            isHighlightingEnabled: true
          } : {})
        }}
        onBarcodeScanned={scanned ? undefined : handleScan}
      >
        <View style={styles.scanOverlay}>
          <View style={[
            styles.targetBox,
            Platform.OS === 'ios' ? styles.iosTargetBox : styles.androidTargetBox
          ]} />
        </View>
      </CameraView>

          {/* Debug overlay */}
          <View style={styles.debugOverlay}>
            <Text style={styles.debugText}>
              {debugMsg || 'Waiting for barcode...'}
            </Text>
          </View>
          <View style={styles.buttons}>
            <Button
              title={`Switch to ${camType === 'back' ? 'front' : 'back'} camera`}
              onPress={() =>
                setCamType(camType === 'back' ? 'front' : 'back')
              }
            />
            <Button
              title="Show Saved Products"
              onPress={() => setShowSavedProducts(true)}
            />
          </View>
        </>
      ) : (
        <>
          <Button
            title="Back to Scanner"
            onPress={() => setShowSavedProducts(false)}
          />
          {savedProducts.length > 0 ? (
            <FlatList
              data={savedProducts}
              keyExtractor={(item) => item.id}
              renderItem={renderSavedProductItem}
              style={{ marginTop: 10 }}
            />
          ) : (
            <View style={styles.noProductsContainer}>
              <Text style={styles.noProductsText}>No saved products yet. Scan some barcodes!</Text>
            </View>
          )}
          {savedProducts.length > 0 && (
            <Button
              title="Clear All Saved Products"
              onPress={clearAllProducts}
              color="red"
            />
          )}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    flex: 1,
  },
  debugOverlay: {
    position: 'absolute',
    bottom: 160,
    width: '100%',
    padding: 8,
    backgroundColor: '#000000cc',
  },
  debugText: {
    color: '#fff',
    textAlign: 'center',
  },
  buttons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 12,
  },
  productItem: {
    padding: 12,
    borderBottomWidth: 1,
    borderColor: '#ccc',
    backgroundColor: '#fff',
    marginHorizontal: 8,
    marginVertical: 4,
    borderRadius: 6,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  productName: {
    fontWeight: 'bold',
    fontSize: 16,
  },
  productCode: {
    fontStyle: 'italic',
    color: '#666',
  },
  productDate: {
    color: '#999',
    fontSize: 12,
    marginTop: 4,
  },
  tapForMore: {
    color: 'blue',
    fontSize: 12,
    marginTop: 4,
    textAlign: 'right',
  },
  // New styles for the form (replacing modal)
  formContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f8f8',
  },
  formHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  formTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
  },
  closeButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 15,
    fontSize: 16,
  },
  formButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  button: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  saveButton: {
    backgroundColor: '#4CAF50',
  },
  cancelButton: {
    backgroundColor: '#f44336',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  detailsContainer: {
    flex: 1,
    padding: 15,
  },
  detailsScroll: {
    flex: 1,
    marginBottom: 10,
  },
  detailsHeader: {
    marginBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    paddingBottom: 15,
  },
  detailsTitle: {
    fontSize: 22,
    fontWeight: 'bold',
  },
  detailsBarcode: {
    fontStyle: 'italic',
    color: '#666',
    marginTop: 5,
  },
  detailsSection: {
    marginBottom: 20,
    backgroundColor: '#f9f9f9',
    padding: 12,
    borderRadius: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  detailRow: {
    flexDirection: 'row',
    marginBottom: 6,
    flexWrap: 'wrap',
  },
  detailLabel: {
    fontWeight: 'bold',
    width: '40%',
  },
  detailValue: {
    flex: 1,
  },
  highLevel: {
    color: 'red',
    fontWeight: 'bold',
  },
  moderateLevel: {
    color: 'orange',
  },
  lowLevel: {
    color: 'green',
  },
  ingredientText: {
    marginBottom: 4,
  },
  rawDataButton: {
    backgroundColor: '#eee',
    padding: 10,
    borderRadius: 5,
    alignItems: 'center',
  },
  rawDataButtonText: {
    color: '#555',
  },
  noProductsContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  noProductsText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  }
});